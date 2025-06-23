import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
import json

CHROMA_PATH = "chroma"
DATA_PATH = "data"
FAQ_PATH = "faqs.json"  # New: Preloaded FAQs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        clear_database()

    # Load documents and FAQs
    documents = load_documents()
    chunks = split_documents(documents)
    faq_chunks = load_faqs() if os.path.exists(FAQ_PATH) else []
    
    # Store in Chroma
    add_to_chroma(chunks + faq_chunks)
    
    # Create BM25 retriever for hybrid search
    create_bm25_index(chunks)

def load_documents():
    print(f"Loading documents from {DATA_PATH}")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=300,
        separators=["\n• ", "\n- ", "\n#", "\n##", "\nFigure", "\nTable"],
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def load_faqs():
    with open(FAQ_PATH) as f:
        faqs = json.load(f)
    return [
        Document(
            page_content=answer,
            metadata={
                "source": "preloaded_faq",
                "page": 0,
                "faq_question": question
            }
        ) for question, answer in faqs.items()
    ]

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_ids = set(db.get()["ids"])
    new_chunks = [c for c in chunks_with_ids if c.metadata["id"] not in existing_ids]

    if new_chunks:
        print(f"Adding {len(new_chunks)} new documents")
        db.add_documents(
            new_chunks,
            ids=[c.metadata["id"] for c in new_chunks]
        )
    else:
        print("No new documents to add")

def create_bm25_index(chunks):
    # Save chunks for hybrid search
    with open("bm25_chunks.json", "w") as f:
        json.dump([{"text": c.page_content, "metadata": c.metadata} for c in chunks], f)

def calculate_chunk_ids(chunks):
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        chunk.metadata["id"] = f"{source}:{page}:{i}"
    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists("bm25_chunks.json"):
        os.remove("bm25_chunks.json")

if __name__ == "__main__":
    main()
