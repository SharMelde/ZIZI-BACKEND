import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# 📁 Directory containing your PDF files
PDF_DIRECTORY = "docs"
# 📂 Directory where the FAISS index will be saved
FAISS_INDEX_PATH = "data/faiss_index"

def load_pdfs(directory):
    """Load and return all PDFs from a folder."""
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return splitter.split_documents(documents)

def embed_and_store(chunks):
    """Embed the document chunks and save the FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(FAISS_INDEX_PATH)
    print(f"💾 Saved FAISS index to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    print("🧠 Starting PDF ingestion...")
    docs = load_pdfs(PDF_DIRECTORY)
    print(f"✅ Loaded {len(docs)} documents.")

    chunks = split_documents(docs)
    print(f"✂️ Chunked into {len(chunks)} pieces.")

    embed_and_store(chunks)
