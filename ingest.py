<<<<<<< HEAD
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_DIRECTORY = "docs"
DB_PATH = "data/faiss_index"

def load_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    vectorstore.save_local(DB_PATH)
    print(f"💾 Saved FAISS index to {DB_PATH}")

if __name__ == "__main__":
    print("🧠 Starting PDF ingestion...")
    documents = load_pdfs(PDF_DIRECTORY)
    print(f"✅ Loaded {len(documents)} documents.")
    chunks = chunk_documents(documents)
    print(f"✂️ Chunked into {len(chunks)} pieces.")
    embed_and_store(chunks)
=======
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_DIRECTORY = "docs"
DB_PATH = "data/faiss_index"

def load_pdfs(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def embed_and_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    vectorstore.save_local(DB_PATH)
    print(f"💾 Saved FAISS index to {DB_PATH}")

if __name__ == "__main__":
    print("🧠 Starting PDF ingestion...")
    documents = load_pdfs(PDF_DIRECTORY)
    print(f"✅ Loaded {len(documents)} documents.")
    chunks = chunk_documents(documents)
    print(f"✂️ Chunked into {len(chunks)} pieces.")
    embed_and_store(chunks)
>>>>>>> 21f128ad591456f412735795fb48f8b4568ea0bb
