"""Investigation Agent — combines SQL numbers, ML predictions, and RAG evidence."""
from dataclasses import dataclass, field


@dataclass
class Investigation:
    question: str
    sql_findings: dict
    ml_findings: dict
    rag_findings: dict
    notes: list = field(default_factory=list)
    drilldown: dict = field(default_factory=dict)   # multi-step 'why did X change' decomposition


def combine(question: str, sql_result, ml_results: list, rag_result, drilldown: dict | None = None) -> Investigation:
    sql_findings = {
        "sql": sql_result.sql,
        "rows": sql_result.rows,
        "rejected_reason": sql_result.rejected_reason,
    }
    ml_findings = {r.model_name: r.summary for r in ml_results}
    rag_findings = {"chunks": rag_result.chunks if rag_result else []}

    notes = []
    if sql_result.rejected_reason:
        notes.append(f"SQL query was rejected by the safety validator: {sql_result.rejected_reason}")
    if sql_result.sql and not sql_result.rows:
        notes.append("No rows returned from the warehouse for this question.")
    if rag_result and not rag_result.chunks:
        notes.append("No supporting business documents were retrieved.")

    return Investigation(
        question=question,
        sql_findings=sql_findings,
        ml_findings=ml_findings,
        rag_findings=rag_findings,
        notes=notes,
        drilldown=drilldown or {},
    )
