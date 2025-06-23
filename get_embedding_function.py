from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text",
        num_ctx=512  # Smaller context for embeddings
    )