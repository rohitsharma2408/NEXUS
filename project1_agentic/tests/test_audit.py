"""Audit log tests. Run in an isolated throw-away database so the real audit table is untouched."""
import json
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

import audit

BASE = os.environ["DATABASE_URL"]
ADMIN = BASE.rsplit("/", 1)[0] + "/postgres"
TEST_DB = "nexus_audit_test"


@pytest.fixture(scope="module")
def audit_db(tmp_path_factory):
    try:
        admin = create_engine(ADMIN, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
            c.execute(text(f"CREATE DATABASE {TEST_DB}"))
    except Exception as e:
        pytest.skip(f"Postgres not reachable: {e}")
    url = BASE.rsplit("/", 1)[0] + f"/{TEST_DB}"
    os.environ["AUDIT_DATABASE_URL"] = url
    audit.FALLBACK_PATH = str(tmp_path_factory.mktemp("audit") / "fallback.jsonl")
    audit.reset_engine()
    audit.ensure_schema()
    yield create_engine(url)
    audit.reset_engine()
    os.environ.pop("AUDIT_DATABASE_URL", None)
    with admin.connect() as c:
        c.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))


def test_events_are_written_with_request_and_actor(audit_db):
    rid = audit.begin_request("alice@corp")
    assert audit.log_event("sql_query", "total revenue?", status="executed",
                           sql="SELECT 1 LIMIT 1", row_count=1) is True
    rows = audit.recent(request_id=rid)
    assert len(rows) == 1
    r = rows[0]
    assert r["actor"] == "alice@corp" and r["event_type"] == "sql_query"
    assert r["detail"]["sql"] == "SELECT 1 LIMIT 1" and r["question"] == "total revenue?"


def test_chain_verifies_after_many_events(audit_db):
    audit.begin_request("bob")
    for i in range(25):
        audit.log_event("ml_call", f"q{i}", n=i, status="ok")
    ok, bad, n = audit.verify_chain()
    assert ok and bad is None and n >= 25


def test_update_and_delete_are_blocked(audit_db):
    with pytest.raises(DBAPIError, match="append-only"):
        with audit_db.begin() as c:
            c.execute(text("UPDATE agent_audit_log SET actor = 'mallory'"))
    with pytest.raises(DBAPIError, match="append-only"):
        with audit_db.begin() as c:
            c.execute(text("DELETE FROM agent_audit_log"))
    with pytest.raises(DBAPIError, match="append-only"):
        with audit_db.begin() as c:
            c.execute(text("TRUNCATE agent_audit_log"))


def test_tampering_by_a_superuser_is_detected(audit_db):
    """Even someone who disables the trigger and edits a row leaves a broken chain."""
    audit.begin_request("carol")
    audit.log_event("sql_query", "sensitive q", status="executed", row_count=3)
    assert audit.verify_chain()[0]
    with audit_db.begin() as c:
        c.execute(text("ALTER TABLE agent_audit_log DISABLE TRIGGER trg_audit_no_mutation"))
        c.execute(text("UPDATE agent_audit_log SET question = 'innocent q' WHERE question = 'sensitive q'"))
        c.execute(text("ALTER TABLE agent_audit_log ENABLE TRIGGER trg_audit_no_mutation"))
    ok, bad_id, _ = audit.verify_chain()
    assert not ok and bad_id is not None


def test_deleted_row_breaks_chain_too(audit_db):
    audit.ensure_schema()
    with audit_db.begin() as c:
        c.execute(text("ALTER TABLE agent_audit_log DISABLE TRIGGER trg_audit_no_mutation"))
        c.execute(text("DELETE FROM agent_audit_log WHERE id = (SELECT min(id) + 2 FROM agent_audit_log)"))
        c.execute(text("ALTER TABLE agent_audit_log ENABLE TRIGGER trg_audit_no_mutation"))
    assert audit.verify_chain()[0] is False


def test_result_values_are_never_stored(audit_db):
    rid = audit.begin_request("dave")
    audit.log_event("sql_query", "q", status="executed", row_count=2, columns=["customer_id"],
                    pii_columns_dropped=["first_name"])
    d = json.dumps(audit.recent(request_id=rid)[0]["detail"])
    assert "first_name" in d and "Laura" not in d


def test_db_failure_falls_back_to_file_and_does_not_raise(tmp_path):
    os.environ["AUDIT_DATABASE_URL"] = "postgresql+psycopg2://nobody:x@localhost:1/none"
    audit.reset_engine()
    audit.FALLBACK_PATH = str(tmp_path / "fb.jsonl")
    audit.begin_request("erin")
    assert audit.log_event("investigation_complete", "q", confidence="low") is False
    line = json.loads(open(audit.FALLBACK_PATH).read().splitlines()[0])
    assert line["event_type"] == "investigation_complete" and line["actor"] == "erin"
    os.environ.pop("AUDIT_DATABASE_URL", None)
    audit.reset_engine()
