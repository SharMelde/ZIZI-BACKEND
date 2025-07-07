from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)

print("✅ FAISS index loaded.")
print("📊 Number of vectors:", db.index.ntotal)

