# Local PDF RAG Chatbot (LangChain + Chroma + Ollama)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on the contents of uploaded PDF files. It uses:

- **LangChain** for orchestration  
- **ChromaDB** for vector storage  
- **Ollama** for running a local LLM (e.g. `llama3`, `gemma:2b`)  
- **FastAPI** for potential future deployment (optional)

---

## Project Structure

PDF_Rag/
data/ # PDF files 

chroma/ # Local ChromaDB vector store

populate_database.py # Loads and embeds PDF data into vector DB

query_data.py # Handles user query and retrieves answers

test_rag.py # Script to run automated benchmark tests

 requirements.txt # All required Python packages

 .gitignore # Files/folders to ignore in version control

 README.md # Project instructions


## Setup Instructions


### 1. Create and Activate a Virtual Environment

<pre> ```bash
  python3 -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate ``` </pre>

### 3. Install Requirements

<pre> ```bash
  pip install -r requirements.txt ``` </pre>


### 4. Pull a Local LLM Model with Ollama
If you haven't already install Ollama then follow this guide. 

<pre> ```bash
  ollama pull phi3:3.8b #model
  ollama pull nomic-embed-text #embedding model
  ``` </pre>

### 5.Load PDF Data into the Vector Store
Put your PDFs inside the data/ folder.

Then run the following command to load and embed the PDFs:
<pre> ```bash
  python populate_database.py --reset #--reset clears the existing ChromaDB and creates a new one. 
  ``` </pre>


