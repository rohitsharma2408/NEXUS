"""
NEXUS Supervisor — classifies intent, routes to SQL/ML/RAG agents in parallel (here,
sequentially for simplicity; swap in LangGraph's parallel node execution for production),
then hands off to Investigation -> Evidence Checker -> Business Analyst.

This module is intentionally framework-light (no hard LangGraph dependency) so it runs
the same way locally and in the container; a LangGraph StateGraph wrapper around
`route()` is straightforward to add for stateful multi-turn investigations.
"""
from dataclasses import dataclass

import sql_agent
import ml_agent
import rag_agent
import investigation_agent
import evidence_checker
import number_verifier
import business_analyst
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


def investigate(question: str) -> dict:
    decision = classify(question)

    sql_result = sql_agent.run(question) if decision.needs_sql else sql_agent.SQLResult(sql="", rows=[], columns=[])

    ml_results = []
    if decision.needs_ml:
        if decision.ml_type == "forecast":
            ml_results.append(ml_agent.forecast_next_month())
        elif decision.ml_type == "churn":
            ml_results.append(ml_agent.get_churn_risk())
        elif decision.ml_type == "anomaly":
            ml_results.append(ml_agent.get_recent_anomalies())
        elif decision.ml_type == "supplier_risk":
            ml_results.append(ml_agent.get_supplier_risk())

    rag_result = None
    if decision.needs_rag:
        try:
            rag_result = rag_agent.retrieve(question)
        except Exception as e:
            rag_result = rag_agent.RAGResult(chunks=[])
            sql_result.rows  # no-op, keep flow going
            print(f"RAG retrieval skipped: {e}")

    investigation = investigation_agent.combine(question, sql_result, ml_results, rag_result)
    evidence = evidence_checker.check(investigation)
    report = business_analyst.write_report(investigation, evidence)

    report["routing"] = {
        "needs_sql": decision.needs_sql,
        "needs_ml": decision.needs_ml,
        "ml_type": decision.ml_type,
        "needs_rag": decision.needs_rag,
    }
    verification = number_verifier.verify_answer(report["answer"], investigation)
    evidence_checker.apply_verification(report, verification)
    report["verification"] = verification
    return report	
