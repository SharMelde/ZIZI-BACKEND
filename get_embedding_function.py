from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text:latest",  # ✅ use correct tag
        num_ctx=512
    )
