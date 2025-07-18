from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import uuid
from rag import get_response
import re


app = FastAPI()

origins = [
    "http://localhost:3000",  # local dev frontend
    "http://127.0.0.1:3000",
    "https://zizi-chatbot.vercel.app",  # replace with deployed frontend if any
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

from typing import Optional

class QueryResponse(BaseModel):
    answer: str
    source: Optional[str] = None
    more_info: Optional[str] = None

history_log = []

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = get_response(request.question)

    # Clean up answer formatting for frontend
    clean_answer = result["answer"].strip()

    # Convert • and numbered lines to Markdown-style dash bullets for web UI
    lines = clean_answer.split("\n")
    formatted_lines = []
    for line in lines:
        if re.match(r'^(\d+[\.\)]|[a-zA-Z][\.\)]|[-•–])\s+', line):
            formatted_lines.append(f"- {re.sub(r'^(\d+[\.\)]|[a-zA-Z][\.\)]|[-•–])\s+', '', line)}")
        else:
            formatted_lines.append(line)
    display_answer = "\n".join(formatted_lines)

    # Save query to in-memory log (later can move to file/db)
    history_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "question": request.question.strip(),
        "answer": display_answer,
        "source": result.get("source")
    })

    return QueryResponse(
        answer=display_answer,
        source=result.get("source"),
        more_info=result.get("more_info")
    )

@app.get("/history")
def get_history():
    return history_log
