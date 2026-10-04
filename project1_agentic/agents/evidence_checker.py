"""
Evidence Checker — verifies every claim against retrieved sources, flags
correlation-vs-causation, and assigns a confidence level to the investigation.

Also verifies the FINAL ANSWER's numbers (verify_grounding, called after the Business
Analyst writes the report) against the actual evidence returned. The pre-answer check()
below can only reason about which sources responded, not what the LLM then wrote — that
gap is exactly how a real bug got through earlier: the analyst relabeled historical
months as a future forecast, and nothing checked the written numbers against the data
that was actually retrieved. verify_grounding() closes that gap.
"""
import re
from dataclasses import dataclass

# Numbers at or below this magnitude, if they're whole numbers, are usually list ranks,
# month counts, or "top N" phrasing rather than data values, and would otherwise produce
# constant false positives (e.g. "here are the top 5 months" flagging "5" as ungrounded).
_SMALL_INT_IGNORE_THRESHOLD = 12


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


def _extract_numbers(text: str) -> list[float]:
    raw = re.findall(r"-?\$?\d[\d,]*\.?\d*%?", text)
    out = []
    for m in raw:
        cleaned = m.replace("$", "").replace(",", "").replace("%", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def _flatten_numbers(obj) -> list[float]:
    nums = []
    if isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_flatten_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            nums.extend(_flatten_numbers(v))
    else:
        try:
            nums.append(float(obj))
        except (TypeError, ValueError):
            pass
    return nums


def verify_grounding(answer_text: str, investigation, tolerance_pct: float = 0.01) -> tuple[list[float], int]:
    """
    Checks every number in the written answer against the numbers actually present in
    the SQL rows and ML findings. Returns (ungrounded_numbers, total_numbers_checked).
    An ungrounded number is one that appears in the prose but nowhere in the evidence —
    exactly the signature of the analyst inventing or relabeling a figure.
    """
    answer_nums = _extract_numbers(answer_text)
    evidence_nums = _flatten_numbers(investigation.sql_findings.get("rows", []))
    evidence_nums += _flatten_numbers(investigation.ml_findings)

    ungrounded = []
    for n in answer_nums:
        if abs(n) <= _SMALL_INT_IGNORE_THRESHOLD and n == int(n):
            continue  # likely a rank, a count like "top 5", or a month/day number
        if n == int(n) and 1900 <= n <= 2100:
            continue  # a calendar year (e.g. "December 2024"), not a data value
        tolerance = max(0.5, abs(n) * tolerance_pct)
        if not any(abs(n - e) <= tolerance for e in evidence_nums):
            ungrounded.append(n)
    return ungrounded, len(answer_nums)
