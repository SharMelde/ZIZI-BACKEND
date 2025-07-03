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

