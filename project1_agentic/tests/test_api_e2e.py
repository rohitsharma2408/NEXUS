"""HTTP-level tests of the real FastAPI app on the real warehouse (LLM stubbed; there is no key in CI).
Covers what unit tests can't: the actor header reaching the audit log, the audit endpoints' auth,
and the response shape (charts, verification)."""
import json
import os
import sys
import uuid

import pytest
from sqlalchemy import create_engine, text

RO = os.environ["READONLY_DATABASE_URL"]
TOKEN = "test-admin-token"


@pytest.fixture(scope="module")
def client():
    try:
        with create_engine(RO).connect() as c:
            c.execute(text("SELECT count(*) FROM fact_sales")).scalar()
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")
    os.environ["AUDIT_ADMIN_TOKEN"] = TOKEN
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def llm(monkeypatch):
    state = {"sql": "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"}

    def fake(system, prompt, max_tokens=1000):
        if system.startswith("Classify"):
            return json.dumps({"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False})
        if "single read-only PostgreSQL" in system:
            return state["sql"]
        payload = json.loads(prompt.split("Evidence bundle (JSON):\n", 1)[1].split("\n\nWrite the final report.")[0])
        dd = payload.get("drilldown") or {}
        if dd:
            return f"Revenue changed {dd['total_change_pct']}% between {dd['baseline_period']} and {dd['target_period']}."
        return f"{len(payload['sql_findings']['rows'])} rows."

    for name in ("supervisor", "business_analyst", "sql_agent"):
        monkeypatch.setattr(sys.modules[name], "call_llm", fake)
    return state


@pytest.fixture(params=["sequential", "langgraph"])
def mode(request, monkeypatch):
    if request.param == "langgraph":
        pytest.importorskip("langgraph")
        monkeypatch.setenv("USE_LANGGRAPH", "1")
    else:
        monkeypatch.delenv("USE_LANGGRAPH", raising=False)
    return request.param


def test_actor_header_reaches_every_audit_event(client, llm, mode):
    actor = f"test-{mode}-{uuid.uuid4().hex[:8]}@nexus"     # unique per run: the audit table persists
    r = client.post("/investigate", json={"question": "Why did revenue fall in February 2023?"},
                    headers={"x-actor": actor})
    assert r.status_code == 200
    rows = client.get("/audit/recent?limit=50", headers={"x-admin-token": TOKEN}).json()
    done = next(e for e in rows if e["event_type"] == "investigation_complete" and e["actor"] == actor)
    same_request = [e for e in rows if e["request_id"] == done["request_id"]]
    kinds = {e["event_type"] for e in same_request}
    assert {"sql_query", "drilldown", "investigation_complete"} <= kinds
    assert {e["actor"] for e in same_request} == {actor}      # every branch kept the caller's identity


def test_response_has_charts_and_verified_numbers(client, llm, mode):
    b = client.post("/investigate", json={"question": "Why did revenue fall in February 2023?"}).json()
    assert b["charts"] and b["verification"]["unsupported"] == [] and b["routing"]["drilldown"] is True


def test_pii_request_returns_no_rows(client, llm, mode):
    llm["sql"] = "SELECT first_name, last_name FROM dim_customer LIMIT 10"
    b = client.post("/investigate", json={"question": "List customer names"}).json()
    assert b["evidence"]["sql"]["rows"] == [] and "not available" in b["evidence"]["sql"]["rejected_reason"]


def test_audit_endpoints_require_the_admin_token(client):
    assert client.get("/audit/verify").status_code == 403
    assert client.get("/audit/verify", headers={"x-admin-token": "wrong"}).status_code == 403
    assert client.get("/audit/recent").status_code == 403
    ok = client.get("/audit/verify", headers={"x-admin-token": TOKEN})
    assert ok.status_code == 200 and ok.json()["chain_intact"] is True


def test_empty_question_rejected(client):
    assert client.post("/investigate", json={"question": "   "}).status_code == 400
