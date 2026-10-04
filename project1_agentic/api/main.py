import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from agents import supervisor  # noqa: E402
import audit_log  # noqa: E402

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
def investigate(req: InvestigateRequest, request: Request):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    started = time.perf_counter()
    user = request.headers.get("x-user-id")
    try:
        result = supervisor.investigate(req.question)
    except Exception as e:
        audit_log.record(req.question, error=e, requested_by=user,
                         latency_ms=(time.perf_counter() - started) * 1000)
        raise HTTPException(status_code=500, detail=str(e))
    audit_log.record(req.question, result=result, requested_by=user,
                     latency_ms=(time.perf_counter() - started) * 1000)
    return result
