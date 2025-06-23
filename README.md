# Local PDF RAG Chatbot (LangChain + Chroma + Ollama)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions based on the contents of uploaded PDF files. It uses:

- **LangChain** for orchestration  
- **ChromaDB** for vector storage  
- **Ollama** for running a local LLM (e.g. `llama3`, `gemma:2b`)  
- **FastAPI** for potential future deployment (optional)

---

## Project Structure

PDF_Rag/
│
├── data/ # PDF files (e.g., reports, publications)
├── chroma/ # Local ChromaDB vector store
├── populate_database.py # Loads and embeds PDF data into vector DB
├── query_data.py # Handles user query and retrieves answers
├── test_rag.py # Script to run automated benchmark tests
├── requirements.txt # All required Python packages
├── .gitignore # Files/folders to ignore in version control
└── README.md # Project instructions


## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
(If you uploaded files manually instead of cloning, just navigate to your project folder in the terminal.)

2. Create and Activate a Virtual Environment
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install Requirements
bash
Copy
Edit
pip install -r requirements.txt
4. Pull a Local LLM Model with Ollama
If you haven't already installed Ollama, follow this guide. Then pull a model:

bash
Copy
Edit
ollama pull llama3
Or for Gemma:

bash
Copy
Edit
ollama pull gemma:2b
Load PDF Data into the Vector Store
Put your PDFs inside the data/ folder.

Then run the following command to load and embed the PDFs:

bash
Copy
Edit
python populate_database.py --reset
--reset clears the existing ChromaDB and creates a new one.

This step will take a few seconds per document depending on size and model.

Ask a Question via CLI
You can run this script and enter your own queries interactively:

bash
Copy
Edit
python query_data.py
Follow the prompt and type in a natural language question (e.g., "How many TVET institutions exist across 47 counties?").

Run Benchmarks (Test RAG Responses)
To run automated test cases (including latency and keyword checks), use:

bash
Copy
Edit
python test_rag.py
This will output test results, pass/fail status, and timing for each query.

📄 Example PDF Used
For testing purposes, the following file is included:

bash
Copy
Edit
data/impact-report.pdf
You can replace or add more PDFs into the data/ folder.

✅ .gitignore
This project ignores virtual environments, cache files, and Chroma artifacts. Here's a sample .gitignore:

gitignore
Copy
Edit
*.pyc
__pycache__/
.DS_Store
venv/
backup/
chroma/
