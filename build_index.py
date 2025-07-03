from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

DATA_FOLDER = "data"
PDF_FILES = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

docs = []
for pdf_file in PDF_FILES:
    loader = PyPDFLoader(os.path.join(DATA_FOLDER, pdf_file))
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

print(f"✅ Loaded {len(PDF_FILES)} PDFs and split into {len(chunks)} chunks")

embeddings = OllamaEmbeddings(model="llama3")
db = FAISS.from_documents(chunks, embeddings)

db.save_local(os.path.join(DATA_FOLDER, "faiss_index"))
print("✅ FAISS index created and saved.")
