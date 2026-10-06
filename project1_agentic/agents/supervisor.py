"""
NEXUS Supervisor — classifies intent, gathers evidence from the SQL / ML / RAG agents and the
multi-step drill-down, then hands off to Investigation -> Evidence Checker -> Business Analyst ->
Number Verifier.

Two execution modes over the SAME step functions (so behaviour cannot drift between them):

  investigate()        plain-Python, sequential. Default; no extra dependency.
  investigate_graph()  LangGraph StateGraph. classify fans out to sql / ml / rag / drilldown in
                       PARALLEL, joins at `combine`, then check -> write -> verify.
                       Enable for the API with USE_LANGGRAPH=1.
"""
import operator
import os
from dataclasses import dataclass
from typing import Annotated, TypedDict

import sql_agent
import ml_agent
import rag_agent
import investigation_agent
import investigation_playbook
import evidence_checker
import number_verifier
import business_analyst
import charts
import audit
import pii
from config import call_llm

INTENT_SYSTEM = """Classify a business question into a JSON object:
{"needs_sql": bool, "needs_ml": bool, "ml_type": "forecast"|"churn"|"anomaly"|"supplier_risk"|null,
 "needs_rag": bool}
needs_sql is almost always true. needs_ml is true if the question is about predictions, risk,
churn, or unusual/anomalous patterns. needs_rag is true if the question could be explained by a
policy, promotion, or operational document (e.g. "why did X change"). Return ONLY the JSON."""


@dataclass
class RouteDecision:
    needs_sql: bool = True
    needs_ml: bool = False
    ml_type: str | None = None
    needs_rag: bool = False


def classify(question: str) -> RouteDecision:
    import json
    raw = call_llm(INTENT_SYSTEM, question, max_tokens=200)
    try:
        data = json.loads(raw)
        return RouteDecision(**data)
    except Exception:
        # Fail safe: always run SQL at minimum.
        return RouteDecision(needs_sql=True, needs_ml=False, ml_type=None, needs_rag=False)


# ------------------------------------------------------------------ shared step functions
def step_sql(question: str, decision: RouteDecision):
    if decision.needs_sql:
        return sql_agent.run(question)
    return sql_agent.SQLResult(sql="", rows=[], columns=[])


_ML_FUNCS = {   # looked up by name at call time so tests/overrides of ml_agent take effect
    "forecast": "forecast_next_month",
    "churn": "get_churn_risk",
    "anomaly": "get_recent_anomalies",
    "supplier_risk": "get_supplier_risk",
}


def step_ml(question: str, decision: RouteDecision) -> list:
    if not (decision.needs_ml and decision.ml_type in _ML_FUNCS):
        return []
    kind = decision.ml_type
    try:
        res = getattr(ml_agent, _ML_FUNCS[kind])()
        audit.log_event("ml_call", question, ml_type=kind, status="ok", model=res.model_name)
        return [res]
    except Exception as e:
        audit.log_event("ml_call", question, ml_type=kind, status="error", reason=str(e)[:300])
        raise


def step_rag(question: str, decision: RouteDecision):
    if not decision.needs_rag:
        return None
    try:
        res = rag_agent.retrieve(question)
        audit.log_event("rag_retrieval", question, status="ok", chunks=len(res.chunks))
        return res
    except Exception as e:
        audit.log_event("rag_retrieval", question, status="error", reason=str(e)[:300])
        print(f"RAG retrieval skipped: {e}")
        return rag_agent.RAGResult(chunks=[])


def step_drilldown(question: str) -> dict:
    """Multi-step 'why did X change between two periods' decomposition; {} if not that kind of question."""
    try:
        res = investigation_playbook.investigate_change(question)
    except Exception as e:                       # a failed drill-down must never block the answer
        audit.log_event("drilldown", question, status="error", reason=str(e)[:300])
        return {}
    if res:
        audit.log_event("drilldown", question, status="ok", metric=res["metric"],
                        baseline=res["baseline_period"], target=res["target_period"])
    return res or {}


def step_finish(question: str, decision: RouteDecision, sql_result, ml_results, rag_result, drilldown) -> dict:
    investigation = investigation_agent.combine(question, sql_result, ml_results, rag_result, drilldown)
    evidence = evidence_checker.check(investigation)
    report = business_analyst.write_report(investigation, evidence)

    report["routing"] = {
        "needs_sql": decision.needs_sql, "needs_ml": decision.needs_ml,
        "ml_type": decision.ml_type, "needs_rag": decision.needs_rag,
        "drilldown": bool(drilldown),
    }
    report["answer"] = pii.scrub_text(report["answer"])   # last line of defence for PII
    verification = number_verifier.verify_answer(report["answer"], investigation)
    evidence_checker.apply_verification(report, verification)
    report["verification"] = verification
    try:
        report["charts"] = charts.build_charts(investigation)
    except Exception as e:                       # charts are an extra; never fail the answer over one
        report["charts"] = []
        print(f"chart generation skipped: {e}")
    audit.log_event(
        "investigation_complete", question, confidence=report["confidence"],
        routing=report["routing"], sql_rows=len(investigation.sql_findings.get("rows") or []),
        sql_rejected=investigation.sql_findings.get("rejected_reason"),
        figures_checked=verification.get("checked", 0),
        unsupported_figures=verification.get("unsupported", []),
        charts=len(report["charts"]),
    )
    return report


# --------------------------------------------------------------------- sequential mode
def investigate_sequential(question: str) -> dict:
    audit.current_request_id()  # every event of this call shares one request id
    decision = classify(question)
    return step_finish(question, decision, step_sql(question, decision), step_ml(question, decision),
                       step_rag(question, decision), step_drilldown(question))


# ----------------------------------------------------------------------- LangGraph mode
class GraphState(TypedDict, total=False):
    question: str
    decision: RouteDecision
    sql_result: object
    ml_results: list
    rag_result: object
    drilldown: dict
    report: dict
    trace: Annotated[list, operator.add]       # parallel branches append; reducer merges them


def build_graph():
    from langgraph.graph import END, START, StateGraph

    def n_classify(s):
        return {"decision": classify(s["question"]), "trace": ["classify"]}

    def n_sql(s):
        return {"sql_result": step_sql(s["question"], s["decision"]), "trace": ["sql"]}

    def n_ml(s):
        return {"ml_results": step_ml(s["question"], s["decision"]), "trace": ["ml"]}

    def n_rag(s):
        return {"rag_result": step_rag(s["question"], s["decision"]), "trace": ["rag"]}

    def n_drill(s):
        return {"drilldown": step_drilldown(s["question"]), "trace": ["drilldown"]}

    def n_combine(s):
        return {"trace": ["combine"]}

    def n_finish(s):
        rep = step_finish(s["question"], s["decision"], s["sql_result"], s.get("ml_results") or [],
                          s.get("rag_result"), s.get("drilldown") or {})
        return {"report": rep, "trace": ["check+write+verify"]}

    g = StateGraph(GraphState)
    for name, fn in [("classify", n_classify), ("sql", n_sql), ("ml", n_ml), ("rag", n_rag),
                     ("drilldown", n_drill), ("combine", n_combine), ("finish", n_finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    for branch in ("sql", "ml", "rag", "drilldown"):      # fan out: these four run in parallel
        g.add_edge("classify", branch)
    g.add_edge(["sql", "ml", "rag", "drilldown"], "combine")   # join: wait for all four
    g.add_edge("combine", "finish")
    g.add_edge("finish", END)
    return g.compile()


_graph = None


def investigate_graph(question: str) -> dict:
    global _graph
    audit.current_request_id()
    if _graph is None:
        _graph = build_graph()
    out = _graph.invoke({"question": question, "trace": []})
    rep = out["report"]
    rep["routing"]["graph_trace"] = out["trace"]
    return rep


def investigate(question: str) -> dict:
    if os.environ.get("USE_LANGGRAPH", "").lower() in ("1", "true", "yes"):
        return investigate_graph(question)
    return investigate_sequential(question)
