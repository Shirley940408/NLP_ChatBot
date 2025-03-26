from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = generate_answer(request.query)
    return {"answer": answer}
