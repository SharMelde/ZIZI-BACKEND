<<<<<<< HEAD
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text:latest", 
        num_ctx=512
    )
=======
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(
        model="nomic-embed-text:latest",  # ✅ use correct tag
        num_ctx=512
    )
>>>>>>> 21f128ad591456f412735795fb48f8b4568ea0bb
