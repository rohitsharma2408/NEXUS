import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from agents import supervisor  # noqa: E402

app = FastAPI(
    title="NEXUS Agentic AI Layer",
    description="AI Business Analyst API — Project 1 of the NEXUS blueprint.",
    version="1.0.0",
)


class InvestigateRequest(BaseModel):
    question: str


class InvestigateResponse(BaseModel):
    answer: str
    confidence: str
    caveats: list[str]
    evidence: dict
    routing: dict
    verification: dict | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = supervisor.investigate(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result
