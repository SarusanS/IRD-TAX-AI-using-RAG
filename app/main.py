from fastapi import FastAPI
from pydantic import BaseModel
from app.query import answer_question

app = FastAPI(title="IRD Tax AI")

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_tax_question(req: QuestionRequest):
    return answer_question(req.question)
