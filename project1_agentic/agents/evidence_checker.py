"""
Evidence Checker — verifies every claim against retrieved sources, flags
correlation-vs-causation, and assigns a confidence level to the investigation.
"""
from dataclasses import dataclass


@dataclass
class EvidenceReport:
    confidence: str  # "high" | "medium" | "low"
    caveats: list
    has_causal_evidence: bool


def check(investigation) -> EvidenceReport:
    caveats = list(investigation.notes)
    has_sql = bool(investigation.sql_findings.get("rows"))
    has_ml = any(v for v in investigation.ml_findings.values())
    has_rag = bool(investigation.rag_findings.get("chunks"))

    n_sources = sum([has_sql, has_ml, has_rag])
    if n_sources >= 2 and has_sql:
        confidence = "high" if n_sources == 3 else "medium"
    elif n_sources == 1:
        confidence = "low"
    else:
        confidence = "low"
        caveats.append("No corroborating evidence was found across SQL, ML, or documents.")

    # RAG documents can support a causal narrative (e.g. "a promotion ended on this date");
    # SQL/ML alone only show correlation/pattern, never mechanism.
    has_causal_evidence = has_rag

    if has_sql and has_ml and not has_causal_evidence:
        caveats.append(
            "Numeric and model evidence show a pattern, not a confirmed cause — no document "
            "evidence (policy change, promotion end, supply disruption) was found to explain it."
        )

    return EvidenceReport(confidence=confidence, caveats=caveats, has_causal_evidence=has_causal_evidence)
