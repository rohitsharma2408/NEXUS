-- Audit trail of what the agent did per request. Written by the API (owner role);
-- the agent's read-only role must not be able to read or change it.
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    requested_by        TEXT,
    question            TEXT NOT NULL,
    routing             JSONB,
    sql_text            TEXT,
    sql_rows            INTEGER,
    sql_rejected_reason TEXT,
    ml_models           TEXT[],
    rag_chunks          INTEGER,
    confidence          TEXT,
    verification        JSONB,
    n_caveats           INTEGER,
    latency_ms          INTEGER,
    error               TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_audit_log_ts ON agent_audit_log (ts DESC);

REVOKE ALL ON agent_audit_log FROM PUBLIC;
REVOKE ALL ON agent_audit_log FROM nexus_readonly;
REVOKE ALL ON SEQUENCE agent_audit_log_id_seq FROM nexus_readonly;
