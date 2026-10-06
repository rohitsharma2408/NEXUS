"""End-to-end supervisor tests on the REAL warehouse with a stubbed LLM, in both execution modes
(sequential and LangGraph). Skipped when the warehouse isn't reachable."""
import json
import os

import pytest
from sqlalchemy import create_engine, text

import audit
import business_analyst
import supervisor

RO = os.environ["READONLY_DATABASE_URL"]


@pytest.fixture(scope="module", autouse=True)
def warehouse():
    try:
        with create_engine(RO).connect() as c:
            c.execute(text("SELECT count(*) FROM fact_sales")).scalar()
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")


@pytest.fixture
def stub(monkeypatch):
    """LLM stub: routes by keyword, and 'writes' an answer that quotes the evidence numbers (plus a
    leaked e-mail so we can prove the PII scrub works)."""
    calls = {"sql": None, "route": None}

    def fake_llm(system, prompt, max_tokens=1000):
        if system.startswith("Classify"):
            return json.dumps(calls["route"])
        if "single read-only PostgreSQL" in system:
            return calls["sql"]
        payload = json.loads(prompt.split("Evidence bundle (JSON):\n", 1)[1].split("\n\nWrite the final report.")[0])
        dd = payload.get("drilldown") or {}
        if dd:
            return (f"Revenue moved {dd['total_change_pct']}% from {dd['baseline_period']} to {dd['target_period']}. "
                    f"Contact jane.doe@example.com for details.")
        rows = payload["sql_findings"]["rows"]
        return f"The first value is {list(rows[0].values())[1]}." if rows else "No data."

    for mod in (supervisor, business_analyst):
        monkeypatch.setattr(mod, "call_llm", fake_llm)
    import sql_agent
    monkeypatch.setattr(sql_agent, "call_llm", fake_llm)
    monkeypatch.setattr(audit, "log_event", lambda *a, **k: True)
    return calls


@pytest.fixture(params=["sequential", "langgraph"])
def run(request):
    if request.param == "langgraph":
        pytest.importorskip("langgraph")
        return supervisor.investigate_graph
    return supervisor.investigate_sequential


def test_why_question_runs_multistep_drilldown_with_verified_numbers(stub, run):
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    rep = run("Why did revenue fall in February 2023?")
    dd = rep["evidence"]["drilldown"]
    assert dd["baseline_period"] == "2023-01" and dd["target_period"] == "2023-02"
    # ground truth straight from the KPI view
    with create_engine(RO).connect() as c:
        jan, feb = [float(r[0]) for r in c.execute(text(
            "SELECT total_revenue FROM kpi_revenue_by_month WHERE month IN ('2023-01-01','2023-02-01') ORDER BY month"))]
    assert dd["baseline_value"] == pytest.approx(jan, abs=0.01) and dd["target_value"] == pytest.approx(feb, abs=0.01)
    assert dd["total_change_pct"] == pytest.approx((feb - jan) / jan * 100, abs=0.01)
    assert rep["verification"]["unsupported"] == [] and rep["verification"]["checked"] >= 1
    assert rep["routing"]["drilldown"] is True
    assert rep["charts"], "a drill-down answer should come with a chart"


def test_pii_in_model_output_is_scrubbed(stub, run):
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    rep = run("Why did revenue fall in February 2023?")
    assert "jane.doe@example.com" not in rep["answer"] and "[email removed]" in rep["answer"]


def test_plain_question_skips_drilldown_and_returns_chart(stub, run):
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    rep = run("Show monthly revenue")
    assert rep["routing"]["drilldown"] is False
    assert rep["charts"][0]["type"] == "line" and len(rep["charts"][0]["x"]) == 36


def test_forecast_route_reports_units_and_category_chart(stub, run):
    if not os.path.exists(os.path.join(os.environ["MODEL_DIR"], "demand_forecast.joblib")):
        pytest.skip("demand_forecast model not trained (run project2_analytics/ml/train_forecasting.py)")
    stub["route"] = {"needs_sql": True, "needs_ml": True, "ml_type": "forecast", "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    rep = run("Forecast next month")
    fc = rep["evidence"]["ml"]["demand_forecast"]
    assert fc["forecast_month"] == "2025-01" and "not revenue" in fc["unit"]
    assert any("Forecast units by category" in c["title"] for c in rep["charts"])


def test_pii_question_returns_no_names(stub, run):
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT first_name, last_name FROM dim_customer LIMIT 10"
    rep = run("List our customers' names")
    assert rep["evidence"]["sql"]["rows"] == [] and rep["evidence"]["sql"]["rejected_reason"]
    blob = json.dumps(rep, default=str)
    assert "Laura" not in blob and "@email.com" not in blob


def test_graph_runs_all_four_branches_and_matches_sequential(stub):
    pytest.importorskip("langgraph")
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    q = "Why did revenue fall in February 2023?"
    g, s = supervisor.investigate_graph(q), supervisor.investigate_sequential(q)
    assert set(g["routing"]["graph_trace"]) >= {"classify", "sql", "ml", "rag", "drilldown", "combine", "check+write+verify"}
    assert g["evidence"]["drilldown"] == s["evidence"]["drilldown"]
    assert g["answer"] == s["answer"] and g["confidence"] == s["confidence"]


def test_every_audit_event_of_one_request_shares_request_id_and_actor(stub, run, monkeypatch):
    """Regression: LangGraph runs branches in worker threads; they must keep the caller's identity."""
    seen = []
    monkeypatch.setattr(audit, "log_event",
                        lambda et, q=None, **d: seen.append((et, audit.current_request_id(), audit._actor.get())) or True)
    stub["route"] = {"needs_sql": True, "needs_ml": False, "ml_type": None, "needs_rag": False}
    stub["sql"] = "SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 36"
    rid = audit.begin_request("rohit@nexus")
    run("Why did revenue fall in February 2023?")
    kinds = {e[0] for e in seen}
    assert {"sql_query", "drilldown", "investigation_complete"} <= kinds
    assert {e[1] for e in seen} == {rid} and {e[2] for e in seen} == {"rohit@nexus"}
