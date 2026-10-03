"""Business Analyst — turns evidence + confidence into the final explanation."""
import json

from config import call_llm

SYSTEM = """You are NEXUS's Business Analyst. You are given a business question and evidence
gathered by SQL, ML, and document-retrieval agents, plus a confidence level and caveats from
an evidence checker. Write a final answer that:
- Directly answers the question
- Cites the specific numbers/evidence used
- Clearly separates correlation from causation
- Do NOT state a confidence level or list caveats; those are attached separately
- Is concise (under 250 words) and written for a business stakeholder, not an engineer
Rules for ML evidence:
- ml_findings.demand_forecast contains next-month predicted UNITS per product, for the top
  10 products only. It is NOT a revenue forecast. Report it as units, and never present it
  as revenue.
- If the question asks for a revenue forecast and no revenue forecast exists in the evidence,
  say so plainly. Do not build one by relabeling or shifting historical months, and do not
  describe past values as predictions.
- Historical SQL rows describe what already happened. Label them with their actual dates.
"""


def write_report(investigation, evidence_report) -> dict:
    payload = {
        "question": investigation.question,
        "sql_findings": investigation.sql_findings,
        "ml_findings": investigation.ml_findings,
        "rag_findings": investigation.rag_findings,
        "confidence": evidence_report.confidence,
        "caveats": evidence_report.caveats,
    }
    prompt = f"Evidence bundle (JSON):\n{json.dumps(payload, default=str, indent=2)}\n\nWrite the final report."
    answer_text = call_llm(SYSTEM, prompt, max_tokens=800)

    return {
        "answer": answer_text,
        "confidence": evidence_report.confidence,
        "caveats": evidence_report.caveats,
        "evidence": {
            "sql": investigation.sql_findings,
            "ml": investigation.ml_findings,
            "rag": investigation.rag_findings,
        },
    }
