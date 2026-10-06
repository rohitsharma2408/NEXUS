"""
Audit log — a tamper-evident record of exactly what the agent did on a person's behalf.

Every SQL attempt (executed, rejected or errored), ML call, document retrieval and finished
investigation is written to `agent_audit_log`:

  * append-only: a trigger rejects UPDATE / DELETE / TRUNCATE,
  * hash-chained: each row stores sha256(prev_hash + its own content); verify_chain() detects any
    edited or removed row,
  * privacy-preserving: stores the question, SQL text, row counts, column names and which PII
    columns were dropped. It never stores result values.
  * never blocks the user: if the database is unreachable the event goes to a local JSONL
    fallback file instead, and the answer is still returned.

Actor / request identity comes from begin_request(), which the API calls per HTTP request.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

DDL = [
    """
    CREATE TABLE IF NOT EXISTS agent_audit_log (
        id          BIGSERIAL PRIMARY KEY,
        ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
        request_id  TEXT NOT NULL,
        actor       TEXT NOT NULL,
        event_type  TEXT NOT NULL,
        question    TEXT,
        detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
        prev_hash   TEXT NOT NULL,
        row_hash    TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_audit_request ON agent_audit_log(request_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_ts ON agent_audit_log(ts)",
    """
    CREATE OR REPLACE FUNCTION agent_audit_log_immutable() RETURNS trigger AS $$
    BEGIN
        RAISE EXCEPTION 'agent_audit_log is append-only';
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS trg_audit_no_mutation ON agent_audit_log",
    """
    CREATE TRIGGER trg_audit_no_mutation BEFORE UPDATE OR DELETE ON agent_audit_log
        FOR EACH ROW EXECUTE FUNCTION agent_audit_log_immutable()
    """,
    "DROP TRIGGER IF EXISTS trg_audit_no_truncate ON agent_audit_log",
    """
    CREATE TRIGGER trg_audit_no_truncate BEFORE TRUNCATE ON agent_audit_log
        FOR EACH STATEMENT EXECUTE FUNCTION agent_audit_log_immutable()
    """,
]

GENESIS = "0" * 64
FALLBACK_PATH = os.environ.get("AUDIT_FALLBACK_PATH", "logs/audit_fallback.jsonl")
MAX_TEXT = 4000

_request_id = contextvars.ContextVar("audit_request_id", default=None)
_actor = contextvars.ContextVar("audit_actor", default="anonymous")
_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        url = os.environ.get("AUDIT_DATABASE_URL") or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("no AUDIT_DATABASE_URL / DATABASE_URL configured")
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def reset_engine() -> None:
    """For tests: forget the cached engine so a changed env var takes effect."""
    global _engine
    _engine = None


def ensure_schema() -> None:
    with _get_engine().begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))


def begin_request(actor: str | None = None) -> str:
    rid = uuid.uuid4().hex
    _request_id.set(rid)
    _actor.set((actor or "anonymous")[:200])
    return rid


def current_request_id() -> str:
    rid = _request_id.get()
    if rid is None:
        rid = begin_request()
    return rid


def _hash(prev: str, payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + blob).encode()).hexdigest()


def _clip(v, n=MAX_TEXT):
    return v[:n] if isinstance(v, str) else v


def log_event(event_type: str, question: str | None = None, **detail) -> bool:
    """Record one event. Returns True if it reached the database, False if it fell back to file."""
    rid, actor = current_request_id(), _actor.get()
    detail = {k: _clip(v) for k, v in detail.items()}
    question = _clip(question, 2000)
    try:
        with _get_engine().begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(727001)"))  # serialise the chain
            prev = conn.execute(
                text("SELECT row_hash FROM agent_audit_log ORDER BY id DESC LIMIT 1")
            ).scalar() or GENESIS
            ts = datetime.now(timezone.utc)
            payload = {"ts": ts.isoformat(), "request_id": rid, "actor": actor,
                       "event_type": event_type, "question": question, "detail": detail}
            conn.execute(
                text("""INSERT INTO agent_audit_log
                        (ts, request_id, actor, event_type, question, detail, prev_hash, row_hash)
                        VALUES (:ts, :rid, :actor, :et, :q, CAST(:d AS JSONB), :prev, :h)"""),
                {"ts": ts, "rid": rid, "actor": actor, "et": event_type, "q": question,
                 "d": json.dumps(detail, default=str), "prev": prev, "h": _hash(prev, payload)},
            )
        return True
    except Exception as e:  # never let auditing take the product down
        try:
            os.makedirs(os.path.dirname(FALLBACK_PATH) or ".", exist_ok=True)
            with open(FALLBACK_PATH, "a") as f:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "request_id": rid,
                                    "actor": actor, "event_type": event_type, "question": question,
                                    "detail": detail, "audit_db_error": str(e)[:200]},
                                   default=str) + "\n")
        except Exception:
            pass
        print(f"[audit] database write failed, wrote to fallback file: {str(e)[:120]}")
        return False


def verify_chain() -> tuple[bool, int | None, int]:
    """Recompute every hash. Returns (ok, first_bad_id, rows_checked)."""
    prev, n = GENESIS, 0
    with _get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, ts, request_id, actor, event_type, question, detail, prev_hash, row_hash "
            "FROM agent_audit_log ORDER BY id")).fetchall()
    for r in rows:
        n += 1
        payload = {"ts": r.ts.astimezone(timezone.utc).isoformat(), "request_id": r.request_id,
                   "actor": r.actor, "event_type": r.event_type, "question": r.question,
                   "detail": r.detail}
        if r.prev_hash != prev or r.row_hash != _hash(prev, payload):
            return False, int(r.id), n
        prev = r.row_hash
    return True, None, n


def recent(limit: int = 50, request_id: str | None = None) -> list[dict]:
    q = "SELECT id, ts, request_id, actor, event_type, question, detail FROM agent_audit_log"
    params: dict = {"n": limit}
    if request_id:
        q += " WHERE request_id = :rid"
        params["rid"] = request_id
    q += " ORDER BY id DESC LIMIT :n"
    with _get_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(text(q), params)]
