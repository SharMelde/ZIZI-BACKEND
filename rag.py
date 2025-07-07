import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

nltk.download("punkt")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCS_FOLDER = "docs"
INDEX_FILE = "faiss_index"

def clean_text(text: str) -> str:
    """
    Clean extracted text by fixing spacing and line breaks for better readability.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{2,}', '\n', text)
    # Replace single newlines inside paragraphs with space
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_or_load_faiss_index():
    if os.path.exists(INDEX_FILE):
        print("💾 Loading existing FAISS index...")
        return FAISS.load_local(
            INDEX_FILE,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

    docs = load_documents(DOCS_FOLDER)
    if not docs:
        raise ValueError("❗ No PDF files found in the 'docs/' folder.")

    print(f"✅ Loaded {len(docs)} raw documents.")
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("❗ No text chunks found after splitting documents.")

    print(f"✂️ Split into {len(split_docs)} chunks.")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Creating FAISS index...")
    db = FAISS.from_documents(split_docs, embedding)
    db.save_local(INDEX_FILE)
    print("✅ FAISS index created and saved.")
    return db

db = create_or_load_faiss_index()

def get_response(query: str) -> dict:
    print(f"\n🔎 Received query: {query}")

    if not query.strip():
        return {"answer": "❗ Please enter a valid query.", "source": None}

    docs = db.similarity_search(query, k=5)
    print(f"📄 Chunks retrieved: {len(docs)}")

    if not docs:
        return {"answer": "❗ Sorry, no relevant information found.", "source": None}

    combined_text = " ".join([doc.page_content for doc in docs])
    sentences = sent_tokenize(combined_text)
    print(f"📝 Sentences extracted: {len(sentences)}")

    cleaned_sentences = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if s.lower().startswith("zizi afrique") or "www." in s.lower():
            continue
        if len(s.split()) >= 6 and s not in seen:
            cleaned_sentences.append(s)
            seen.add(s)

    print(f"✅ Clean sentences retained: {len(cleaned_sentences)}")
    if not cleaned_sentences:
        return {"answer": "❗ No useful sentences found.", "source": None}

    sentence_embeddings = sentence_model.encode(cleaned_sentences)
    query_embedding = sentence_model.encode(query)

    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)

    top_sentences = [cleaned_sentences[i] for i in sorted_indices[:2]]
    final_answer = "\n".join(f"- {s}" for s in top_sentences).strip()

    # Clean final answer text for better formatting
    final_answer = clean_text(final_answer)

    metadata = docs[0].metadata
    source_name = metadata.get("source", "Unknown document").split("\\")[-1]
    page_number = metadata.get("page", "Unknown page")
    source_text = f"{source_name} — Page {page_number}"

    print(f"✅ Final answer: {final_answer}")
    print(f"🔗 Source: {source_text}")

    return {
        "answer": final_answer or "❗ Sorry, I couldn't find a good answer.",
        "source": source_text
    }
