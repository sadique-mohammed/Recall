"""FastAPI service exposing the RAG pipeline."""

from fastapi import FastAPI
from pydantic import BaseModel

from rag.generate import answer_question

app = FastAPI(title="RAG Evaluation Pipeline")


class AskRequest(BaseModel):
    question: str
    k: int = 3


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer, sources = answer_question(request.question, request.k)
    return AskResponse(answer=answer, sources=sources)
