from rag import db

print("🔍 Checking documents stored in FAISS index...")

docs = db.similarity_search("4IR", k=5)

if not docs:
    print("❌ No matches found for '4IR'")
else:
    print(f"✅ Found {len(docs)} matches:")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(doc.page_content[:500])  # print first 500 chars
