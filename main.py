from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_response, get_history, clear_history, store_feedback

app = FastAPI()

# CORS Configuration (allow all for dev — restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["http://localhost:3000"] or actual domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models --- #

class QueryRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    source: str
    feedback: str  # "thumbs_up" or "thumbs_down"

# --- Routes --- #

@app.post("/query")
def query(request: QueryRequest):
    """
    Receive a question and return an answer with sources.
    """
    result = get_response(request.question)
    return result  # Dict with {"answer": ..., "sources": ...}

@app.get("/history")
def history():
    """
    Return the saved list of past questions/answers.
    """
    return get_history()

@app.post("/clear-history")
def clear():
    """
    Delete all chatbot history.
    """
    clear_history()
    return {"message": "History cleared"}

@app.post("/feedback")
def feedback(request: FeedbackRequest):
    """
    Store user feedback on an answer.
    """
    store_feedback(
        query=request.query,
        answer=request.answer,
        source=request.source,
        feedback=request.feedback
    )
    return {"message": "Feedback recorded"}
