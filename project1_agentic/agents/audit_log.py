"""
Audit log — records what the agent did for each request: the question, the SQL it ran,
how many rows came back, routing, confidence and number-verification outcome, latency.

Design rules:
  * Never raises. A logging failure must not break an answer; it is printed to stderr.
  * Stores NO result rows and NO answer text (they can contain customer data). The SQL
    text plus a row count is enough to reconstruct what was asked of the warehouse.
  * Writes with DATABASE_URL (owner). The agent's own role (nexus_readonly) has no access
    to this table, so the agent cannot read or alter its own audit trail.
  * `requested_by` comes from an optional X-User-Id header. The API has no authentication,
    so this is self-reported and NOT a verified identity.
"""
import json
import os
import sys

from sqlalchemy import create_engine, text

_engine = None

_INSERT = text("""
    INSERT INTO agent_audit_log
        (requested_by, question, routing, sql_text, sql_rows, sql_rejected_reason,
         ml_models, rag_chunks, confidence, verification, n_caveats, latency_ms, error)
    VALUES
        (:requested_by, :question, CAST(:routing AS jsonb), :sql_text, :sql_rows,
         :sql_rejected_reason, :ml_models, :rag_chunks, :confidence,
         CAST(:verification AS jsonb), :n_caveats, :latency_ms, :error)
""")


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True,
                                pool_size=2, max_overflow=2)
    return _engine


def record(question, result=None, error=None, latency_ms=None, requested_by=None):
    try:
        result = result or {}
        ev = result.get("evidence") or {}
        sql = ev.get("sql") or {}
        ver = result.get("verification") or {}
        params = {
            "requested_by": (requested_by or None) and requested_by[:200],
            "question": (question or "")[:2000],
            "routing": json.dumps(result.get("routing")) if result.get("routing") else None,
            "sql_text": (sql.get("sql") or "")[:4000] or None,
            "sql_rows": len(sql.get("rows") or []) if sql.get("sql") else None,
            "sql_rejected_reason": (sql.get("rejected_reason") or None) and sql["rejected_reason"][:1000],
            "ml_models": list((ev.get("ml") or {}).keys()),
            "rag_chunks": len((ev.get("rag") or {}).get("chunks") or []),
            "confidence": result.get("confidence"),
            "verification": json.dumps({"checked": ver.get("checked"),
                                        "unsupported": ver.get("unsupported", [])}) if ver else None,
            "n_caveats": len(result.get("caveats") or []),
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
            "error": (str(error)[:1000] if error else None),
        }
        with _get_engine().begin() as conn:
            conn.execute(_INSERT, params)
    except Exception as e:  # noqa: BLE001 - logging must never break the request
        print(f"audit log write failed: {e}", file=sys.stderr)
