# === START OF rag.py ===
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import fitz  # PyMuPDF

nltk.download("punkt")

DOCS_FOLDER = "docs"
INDEX_FILE = "faiss_index"
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Grants Intern\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)

def clean_sentence(sentence: str) -> str:
    return re.sub(r"^[\s\-–•]*([a-zA-Z0-9]{1,2})[\.\)]\s+", "", sentence).strip()

def find_links(text: str) -> list:
    return re.findall(r'(https?://[^\s)]+)', text)

def get_enriched_lines(sentences, keywords):
    enriched = []
    for sent, meta in sentences:
        if any(kw.lower() in sent.lower() for kw in keywords):
            enriched.append((sent, meta))
    return enriched

def ocr_page(image):
    return pytesseract.image_to_string(image)

def convert_page_to_image(pdf_path, page_number):
    images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
    return images[0] if images else None

def summarize_table_lines(lines):
    summaries = []
    for line in lines:
        if re.search(r'\d', line) and re.search(r'(children|youth|teachers?|parents?|partners?|officials?|beneficiaries|reached)', line, re.I):
            line = re.sub(r'\s+', ' ', line.strip())
            summaries.append(line)
    return "\n".join(summaries)

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)
        print(f"📄 Processing: {file_path}")
        year_match = re.search(r'20\d{2}', filename)
        year_tag = year_match.group(0) if year_match else "unknown"

        try:
            doc = fitz.open(file_path)
            for page_number in range(len(doc)):
                page = doc.load_page(page_number)
                text = clean_text(page.get_text())
                metadata = {
                    'source': file_path,
                    'page': str(page_number + 1),
                    'year': year_tag
                }

                if not text or len(text.strip()) < 20:
                    print(f"🔍 Page {page_number + 1}: Low content — trying OCR...")
                    image = convert_page_to_image(file_path, page_number)
                    if image:
                        raw_ocr = ocr_page(image)
                        lines = [line.strip() for line in raw_ocr.split('\n') if line.strip()]
                        summary = summarize_table_lines(lines)
                        merged = clean_text("\n".join(lines))
                        text = f"{summary}\n\n{merged}" if summary else merged
                        metadata["ocr"] = True

                if len(text.split()) < 10:
                    continue

                documents.append(
                    type('SimpleDoc', (), {
                        'page_content': text,
                        'metadata': metadata
                    })()
                )
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
    return documents

def create_or_load_faiss_index():
    if os.path.exists(INDEX_FILE):
        print("💾 Loading existing FAISS index...")
        return FAISS.load_local(
            INDEX_FILE,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

    docs = load_documents(DOCS_FOLDER)
    if not docs:
        raise ValueError("❗ No PDF files found in the 'docs/' folder.")

    print(f"✅ Loaded {len(docs)} cleaned documents.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    if not split_docs:
        raise ValueError("❗ No text chunks found after splitting documents.")

    print(f"✂️ Split into {len(split_docs)} chunks.")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Creating FAISS index...")
    db = FAISS.from_documents(split_docs, embedding)
    db.save_local(INDEX_FILE)
    print("✅ FAISS index created and saved.")
    return db

db = create_or_load_faiss_index()

def get_response(query: str) -> dict:
    print(f"\n🔎 Received query: {query}")
    if not query.strip():
        return {"answer": "❗ Please enter a valid query.", "source": None, "more_info": None}

    all_docs = db.similarity_search(query, k=20)
    print(f"📄 Chunks retrieved: {len(all_docs)}")

    question_years = re.findall(r"20\d{2}", query)
    target_year = question_years[0] if question_years else None
    print(f"🎯 Target year from question: {target_year}")

    if target_year:
        year_docs = [d for d in all_docs if d.metadata.get("year") == target_year]
        print(f"📅 Matching year chunks: {len(year_docs)}")
    else:
        year_docs = all_docs[:10]

    def score_chunk(doc):
        score = 0
        content = doc.page_content.lower()
        if 'tenda wema' in content and 'kes' in content:
            score += 20
        if re.search(r'\bKES\b|\bUSD\b|\d{5,}|\d+%|\bshillings?\b', content, re.IGNORECASE):
            score += 10
        if target_year and target_year in content:
            score += 5
        if doc.metadata.get("ocr"):
            score += 2
        if re.search(r'(grade.level|proficiency|competency)', content):
            score += 8
        if re.search(r'(pillar|strategic (focus|area|priority))', content) and re.search(r'(education|inclusion|policy|equity|data)', content):
            score += 12
        if re.search(r'(action point|priority|objective|2023.?2025|consolidate|innovate|engage)', content):
            score += 15
        return score

    scored_docs = sorted(year_docs, key=score_chunk, reverse=True)
    top_docs = scored_docs if scored_docs else year_docs[:5]

    # Extract bullets or numbered resolutions
    if re.search(r'\b(resolution|outcome|agreed|conclusion|result|decision|recommendation|commitment|goal)\b', query, re.I):
        for doc in top_docs:
            lines = doc.page_content.splitlines()
            bullet_lines = [line.strip() for line in lines if line.strip().startswith("•")]
            numbered_lines = [line.strip() for line in lines if re.match(r'^\s*(\d{1,2}[\.\)])\s+', line.strip())]

            if bullet_lines or numbered_lines:
                formatted = "\n".join(bullet_lines or numbered_lines)
                source_name = doc.metadata.get("source", "Unknown").split("\\")[-1]
                page_number = doc.metadata.get("page", "Unknown")
                return {
                    "answer": f"The following resolutions were highlighted:\n{formatted}",
                    "source": f"{source_name} — Page {page_number}",
                    "more_info": None
                }

    all_sentences = []
    fallback_numeric = []
    seen = set()

    for doc in top_docs:
        for sent in sent_tokenize(doc.page_content):
            s_clean = clean_sentence(sent.strip())
            if not s_clean or s_clean.lower().startswith("zizi afrique") or s_clean in seen:
                continue
            seen.add(s_clean)
            if len(s_clean.split()) >= 3:
                all_sentences.append((s_clean, doc.metadata))
            elif re.search(r'\bKES\b|\d{5,}', s_clean):
                fallback_numeric.append((s_clean, doc.metadata))

    if not all_sentences and fallback_numeric:
        print(f"💰 Fallback to numeric-only ({len(fallback_numeric)} lines)")
        all_sentences = fallback_numeric
    elif fallback_numeric:
        all_sentences += fallback_numeric

    if not all_sentences:
        return {"answer": "❗ No useful sentences found.", "source": None, "more_info": None}

    sentences = [s for s, _ in all_sentences]
    embeddings = sentence_model.encode(sentences)
    query_embedding = sentence_model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    sorted_indices = similarities.argsort(descending=True)
    top_lines = [sentences[i] for i in sorted_indices[:5]]

    # Thematic enrichment
    if re.search(r'grade.?level|proficiency|competency', query, re.I):
        percent_lines = [s for s in sentences if re.search(r'\d{1,3}%|\d{1,3}\.\d+%', s)]
        percent_lines = [s for s in percent_lines if re.search(r'proficien|grade.?level|competenc', s, re.I)]
        if percent_lines:
            top_lines = percent_lines[:2] + top_lines

    elif re.search(r'(strategic|pillar|focus area|priority)', query, re.I):
        enriched = get_enriched_lines(all_sentences, ["pillar", "focus", "strategy", "priority", "education", "equity", "policy", "inclusion"])
        enriched_lines = [s for s, _ in enriched]
        if enriched_lines:
            top_lines = enriched_lines[:2] + top_lines

    elif re.search(r'(action point|objective|2023.?2025|mission|vision)', query, re.I):
        enriched = get_enriched_lines(all_sentences, ["action point", "2023–2025", "mission", "consolidate", "engage", "innovate"])
        enriched_lines = [s for s, _ in enriched]
        if enriched_lines:
            top_lines = enriched_lines[:3] + top_lines

    amount_line = next((line for line in top_lines if re.search(r'KES\s?[\d,.]+', line)), "")
    reach_line = next((line for line in top_lines if re.search(r'\b\d{1,3}(,\d{3})*\b.*?(children|youth|learners|beneficiaries)', line, re.I)), "")
    joined = f"{amount_line.strip()} {reach_line.strip()}".strip() if amount_line or reach_line else clean_text(" ".join(top_lines[:2]))

    top_meta = all_sentences[sorted_indices[0]][1]
    source_name = top_meta.get("source", "Unknown").split("\\")[-1]
    page_number = top_meta.get("page", "Unknown")
    source_text = f"{source_name} — Page {page_number}"

    links = []
    for i in sorted_indices[:5]:
        links += find_links(sentences[i])
    links = list(set(links))

    return {
        "answer": joined or "❗ Sorry, I couldn't find a good answer.",
        "source": source_text,
        "more_info": links[0] if links else None
    }

# === END OF rag.py ===
