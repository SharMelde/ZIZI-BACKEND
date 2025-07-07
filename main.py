<<<<<<< HEAD
from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_response
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    source: str
    feedback: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    return {"response": get_response(request.query)}

@app.post("/feedback")
def feedback_endpoint(request: FeedbackRequest):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("feedback_logs.txt", "a", encoding="utf-8") as f:
        f.write(
            f"[{timestamp}]\n"
            f"Query: {request.query}\n"
            f"Answer: {request.answer}\n"
            f"Source: {request.source}\n"
            f"Feedback: {request.feedback}\n\n"
        )
    return {"message": "✅ Feedback logged successfully."}
=======
# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to ["http://localhost:3000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    return {"response": get_response(request.query)}

>>>>>>> 21f128ad591456f412735795fb48f8b4568ea0bb
