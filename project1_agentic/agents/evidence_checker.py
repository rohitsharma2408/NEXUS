"""
Evidence Checker — verifies every claim against retrieved sources, flags
correlation-vs-causation, and assigns a confidence level to the investigation.

Two stages:
  check()               before the answer is written: how many sources responded
  apply_verification()  after the answer is written: are the answer's numbers in the evidence
"""
import re
from dataclasses import dataclass

_CAUSAL_QUESTION = re.compile(
    r"\b(why|cause|caused|causes|reason|reasons|driver|drivers|due to|because|explain|behind)\b", re.I
)


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

    has_drill = bool(getattr(investigation, "drilldown", None))
    n_sources = sum([has_sql, has_ml, has_rag, has_drill])
    if n_sources >= 2 and has_sql:
        confidence = "high" if n_sources >= 3 else "medium"
    elif n_sources == 1:
        confidence = "low"
    else:
        confidence = "low"
        caveats.append("No corroborating evidence was found across SQL, ML, or documents.")

    # RAG documents can support a causal narrative (e.g. "a promotion ended on this date");
    # SQL/ML alone only show correlation/pattern, never mechanism.
    has_causal_evidence = has_rag

    # A drill-down says WHERE the change happened (which country/category/channel) and whether
    # the same move occurs every year; neither is a mechanism.
    seas = (getattr(investigation, "drilldown", None) or {}).get("seasonality") or {}
    if seas.get("consistent_with_seasonality"):
        caveats.append("The same month-to-month move happened in other years, so this is likely "
                       "seasonal rather than a new problem.")
    elif has_drill and not has_causal_evidence and _CAUSAL_QUESTION.search(investigation.question or ""):
        caveats.append("The drill-down shows where the change happened, not why: no document "
                       "evidence of a cause (promotion end, outage, supply issue) was found.")

    # Only relevant when the person is actually asking about causes.
    if (has_sql and has_ml and not has_causal_evidence
            and _CAUSAL_QUESTION.search(investigation.question or "")):
        caveats.append(
            "Numeric and model evidence show a pattern, not a confirmed cause — no document "
            "evidence (policy change, promotion end, supply disruption) was found to explain it."
        )

    return EvidenceReport(confidence=confidence, caveats=caveats, has_causal_evidence=has_causal_evidence)


def apply_verification(report: dict, verification: dict) -> None:
    """Adjust confidence/caveats in place using the number-verification result.

    - Any unsupported figure: cap confidence at "low" and name the figures.
    - All checked figures traced to evidence, and some evidence exists: a "low" rating that
      came only from having a single source is raised to "medium". Never to "high" here;
      "high" still requires SQL + ML + documents.
    """
    unsupported = verification.get("unsupported", [])
    if unsupported:
        report["confidence"] = "low"
        report["caveats"].append(
            "These figures in the answer could not be traced to the evidence: "
            + ", ".join(unsupported)
        )
        return

    ev = report.get("evidence", {})
    has_data = (
        bool(ev.get("sql", {}).get("rows"))
        or any(ev.get("ml", {}).values())
        or bool(ev.get("rag", {}).get("chunks"))
    )
    if verification.get("checked", 0) > 0 and has_data and report.get("confidence") == "low":
        report["confidence"] = "medium"
