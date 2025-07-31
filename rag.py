import os
import json
import time
import fitz  # PyMuPDF
import pytesseract
import re
from PIL import Image
from pdf2image import convert_from_path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# ------------------ CONSTANTS ------------------ #
INDEX_FILE = "faiss_index"
DOCS_FOLDER = "docs"
HISTORY_FILE = "history.json"
FEEDBACK_FILE = "feedback.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

year_pattern = re.compile(r"\b(20\d{2})\b")
title_pattern = re.compile(r"^[A-Z][A-Za-z\s\-\–\:\’\,]{3,40}$")
numeric_pattern = re.compile(r"\b[\d{1,3},]*\d+\b|KES|USD|%")
summary_pattern = re.compile(r"([\d,]+)\s+(children|teachers|youth|parents|KES|USD|shillings|percent|%)", re.IGNORECASE)

# ------------------ PDF LOADING ------------------ #

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text
    doc.close()
    return text if len(text.strip()) > 50 else ocr_pdf(file_path)

def ocr_pdf(file_path):
    images = convert_from_path(file_path, dpi=300)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# ------------------ CLEANING ------------------ #

def clean_chunk(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    lines = [line for line in lines if not title_pattern.match(line)]
    return " ".join(lines)

def extract_year_from_filename(name):
    match = year_pattern.search(name)
    return match.group(1) if match else "0000"

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(DOCS_FOLDER, filename)
            raw = extract_text_from_pdf(path)
            cleaned = clean_chunk(raw)
            docs.append(Document(
                page_content=cleaned,
                metadata={"source": filename, "year": extract_year_from_filename(filename)}
            ))
    return docs

def build_or_load_vectorstore(force_refresh=False):
    if os.path.exists(f"{INDEX_FILE}/index.faiss") and not force_refresh:
        return FAISS.load_local(INDEX_FILE, embedder, allow_dangerous_deserialization=True)
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = clean_chunk(chunk.page_content)
    vs = FAISS.from_documents(chunks, embedder)
    vs.save_local(INDEX_FILE)
    return vs

# ------------------ LOGGING ------------------ #

def log_history(question, answer):
    entry = {
        "question": question,
        "answer": answer,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            pass
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def store_feedback(query, answer, source, feedback):
    entry = {
        "question": query,
        "answer": answer,
        "source": source,
        "feedback": feedback,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    data = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
        except:
            pass
    data.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# ------------------ ANSWERING ------------------ #

def extract_query_year(query):
    matches = re.findall(year_pattern, query)
    return matches if matches else None

def get_most_recent_year(docs):
    years = [int(doc.metadata.get("year", "0")) for doc in docs]
    return str(max(years)) if years else None

def format_sources(docs):
    seen = set()
    result = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            result.append(src)
    return "\nSources:\n" + "\n".join(result)

def is_numeric_question(q):
    return any(word in q.lower() for word in ["how many", "how much", "total", "amount", "number", "percentage", "income", "budget", "revenue", "raised"])

def extract_best_sentences(text, question, top_k=3, year_hint=None):
    lines = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    scored = []
    keywords = [w.lower() for w in question.lower().split() if len(w) > 2]

    for line in lines:
        line_lower = line.lower()
        score = 0
        if numeric_pattern.search(line):
            score += 3
        if any(kw in line_lower for kw in keywords):
            score += 2
        if year_hint and year_hint in line:
            score += 2
        if len(line.split()) > 100:
            score -= 1
        if score > 0:
            scored.append((line.strip(), score))

    scored.sort(key=lambda x: -x[1])
    return [s[0] for s in scored[:top_k]]

def extract_summary_number(sentences):
    for s in sentences:
        match = summary_pattern.search(s)
        if match:
            return f"{match.group(1).replace(',', '')} {match.group(2).capitalize()}"
    return None

def clean_response(answer, sources, query):
    # Remove incorrect numeric summary
    lines = answer.strip().splitlines()
    if len(lines) > 0 and re.match(r"^\d+ [A-Za-z%]+$", lines[0]):
        # Only keep if the line is clearly linked to the query
        if not is_numeric_question(query):
            lines = lines[1:]

    # Clean up spacing and duplication
    clean_sources = []
    seen = set()
    for s in sources.strip().splitlines():
        if s and s not in seen and "Sources" not in s:
            seen.add(s)
            clean_sources.append(s)
    return "\n".join(lines).strip(), "\nSources:\n" + "\n".join(clean_sources)

def answer_query(query):
    vs = build_or_load_vectorstore()
    query_years = extract_query_year(query)
    base_results = vs.similarity_search_with_score(query, k=10)

    filtered = []
    for doc, score in base_results:
        doc_year = doc.metadata.get("year", "0000")
        if query_years and doc_year in query_years:
            filtered.append((doc, score))

    if not filtered and not query_years:
        all_docs = vs.similarity_search_with_score("Zizi Afrique", k=50)
        most_recent_year = get_most_recent_year([doc for doc, _ in all_docs])
        for doc, score in base_results:
            if doc.metadata.get("year") == most_recent_year:
                filtered.append((doc, score))

    if not filtered:
        return {"answer": "❗ No relevant content found for the requested year.", "sources": "None"}

    filtered.sort(key=lambda x: x[1])
    top_docs = [doc for doc, _ in filtered[:3]]

    summary = ""
    if is_numeric_question(query):
        sents = []
        for doc in top_docs:
            sents += extract_best_sentences(doc.page_content, query, year_hint=query_years[0] if query_years else None)
        summary = extract_summary_number(sents)
        answer = (summary + "\n" if summary else "") + "\n".join(sents).strip()
        if not sents:
            answer = "❗ No numeric or specific match found."
    else:
        paras = [doc.page_content[:500] for doc in top_docs if doc.page_content.strip()]
        answer = "\n\n".join(paras).strip() if paras else "❗ No useful content found."

    raw_sources = format_sources(top_docs)
    clean_ans, clean_src = clean_response(answer, raw_sources, query)
    return {"answer": clean_ans, "sources": clean_src}

def get_response(query):
    result = answer_query(query)
    log_history(query, result["answer"])
    return result
