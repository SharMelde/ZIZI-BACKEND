import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load the FAISS index from disk
DB_PATH = os.path.join("data", "faiss_index")
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"FAISS index not found at path: {DB_PATH}")

# Initialize the embedding and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# Set up retriever and LLM
retriever = vectorstore.as_retriever()
llm = Ollama(model="llama3")

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Main function called from FastAPI
def query_rag(question: str) -> str:
    print("🔍 Asking RAG:", question)
    response = qa_chain.invoke({"query": question})
    answer = response.get("result")

    # Optional: Show sources used
    sources = response.get("source_documents", [])
    if sources:
        print("📄 Sources used:")
        for doc in sources:
            print("-", doc.metadata.get("source", "unknown"), f"(pg{doc.metadata.get('page', '?')})")

    return answer or "No answer was generated."
