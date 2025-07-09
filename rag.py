import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

nltk.download("punkt")

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

DOCS_FOLDER = "docs"
INDEX_FILE = "faiss_index"

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)

def clean_sentence(sentence: str) -> str:
    return re.sub(r"^[\s\-–•]*([a-zA-Z0-9]{1,2})[\.\)]\s+", "", sentence).strip()

def remove_boilerplate(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if any(x in line.lower() for x in ["www.ziziafrique", "info@ziziafrique", "follow us", "annual report", "contents"]):
            continue
        if re.fullmatch(r"[0-9\s\W]+", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {file_path}")
            loader = PyMuPDFLoader(file_path)
            pages = loader.load()
            for page in pages:
                page.page_content = remove_boilerplate(page.page_content)
            documents.extend(pages)
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

    # Detect expected year from the query
    expected_year = None
    known_years = ["2020", "2021", "2022", "2023", "2024"]
    for y in known_years:
        if y in query:
            expected_year = y
            break

    # Heuristic: if asking about "future", "beyond", "next year" etc., set default to 2023+
    future_keywords = ["2024", "beyond", "future", "next year", "vision", "looking ahead"]
    if any(k in query.lower() for k in future_keywords):
        expected_year = "2023"

    docs = db.similarity_search(query, k=5)
    print(f"📄 Chunks retrieved: {len(docs)}")

    # Prefer docs from expected year and newer (e.g., 2023+ for "2024 goals")
    if expected_year:
        try:
            year = int(expected_year)
            docs = [
                d for d in docs
                if any(str(y) in d.metadata.get("source", "") for y in range(year, 2031))
            ] or docs
        except:
            pass

    if not docs:
        return {"answer": "❗ Sorry, no relevant information found.", "source": None}

    combined_text = " ".join([doc.page_content for doc in docs])
    sentences = sent_tokenize(combined_text)
    print(f"📝 Sentences extracted: {len(sentences)}")

    cleaned_sentences = []
    seen = set()
    for s in sentences:
        s = s.strip()
        if "www." in s.lower() or "http" in s.lower():
            continue
        if "@" in s or "Follow us" in s or "Annual Report" in s or "Contents" in s:
            continue
        if len(re.sub(r'[^a-zA-Z]', '', s)) < 20:
            continue
        if len(s.split()) >= 4 and s not in seen:
            cleaned_sentences.append(s)
            seen.add(s)

    if not cleaned_sentences:
        print("⚠️ Fallback: Using raw chunk content.")
        fallback = clean_text(sentences[0]) if sentences else ""
        redirect = "You can find more in the full report at https://www.ziziafrique.org"
        return {
            "answer": f"{fallback}\n\n{redirect}".strip(),
            "source": docs[0].metadata.get("source", "Unknown source")
        }

    print(f"✅ Clean sentences retained: {len(cleaned_sentences)}")

    sentence_embeddings = sentence_model.encode(cleaned_sentences)
    query_embedding = sentence_model.encode(query)

    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)

    top_sentences = [clean_sentence(cleaned_sentences[i]) for i in sorted_indices[:3]]
    final_answer = " ".join(top_sentences).strip()
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







