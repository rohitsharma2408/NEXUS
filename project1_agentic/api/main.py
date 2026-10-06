import sys
from pathlib import Path

import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

# Import via the bare module names (agents/ is on sys.path) — the same names the agents use between
# themselves. `from agents import audit` would load a SECOND copy of the module, with its own
# request-context variables, and the actor header set here would never reach the agents' log calls.
import audit  # noqa: E402
import supervisor  # noqa: E402

app = FastAPI(
    title="NEXUS Agentic AI Layer",
    description="AI Business Analyst API — Project 1 of the NEXUS blueprint.",
    version="1.0.0",
)


@app.on_event("startup")
def _init_audit():
    try:
        audit.ensure_schema()
    except Exception as e:  # the API must still start; events then go to the fallback file
        print(f"[audit] could not initialise audit table: {e}")


class InvestigateRequest(BaseModel):
    question: str


class InvestigateResponse(BaseModel):
    answer: str
    confidence: str
    caveats: list[str]
    evidence: dict
    routing: dict
    verification: dict | None = None
    charts: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest, x_actor: str | None = Header(default=None)):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    audit.begin_request(x_actor)
    try:
        result = supervisor.investigate(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


def _require_admin(token: str | None):
    expected = os.environ.get("AUDIT_ADMIN_TOKEN")
    if not expected or token != expected:
        raise HTTPException(status_code=403, detail="audit access requires a valid admin token")


@app.get("/audit/verify")
def audit_verify(x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    ok, bad_id, n = audit.verify_chain()
    return {"chain_intact": ok, "first_bad_id": bad_id, "rows_checked": n}


@app.get("/audit/recent")
def audit_recent(limit: int = 50, request_id: str | None = None,
                 x_admin_token: str | None = Header(default=None)):
    _require_admin(x_admin_token)
    return audit.recent(min(max(limit, 1), 500), request_id)
