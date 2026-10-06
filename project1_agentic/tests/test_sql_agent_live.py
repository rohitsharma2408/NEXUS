"""End-to-end SQL agent tests against the REAL warehouse (read-only role) with a stubbed LLM.
Skipped automatically when Postgres / the warehouse is not available."""
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

import audit
import sql_agent

RO = os.environ["READONLY_DATABASE_URL"]


@pytest.fixture(scope="module", autouse=True)
def warehouse():
    try:
        with create_engine(RO).connect() as c:
            c.execute(text("SELECT count(*) FROM fact_sales")).scalar()
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")


@pytest.fixture
def stub_sql(monkeypatch):
    def _set(*sqls):
        it = iter(sqls)
        monkeypatch.setattr(sql_agent, "generate_sql", lambda q, retry_error=None: next(it))
    return _set


@pytest.fixture
def events(monkeypatch):
    log = []
    monkeypatch.setattr(audit, "log_event", lambda et, q=None, **d: log.append((et, q, d)) or True)
    return log


# ---- DB-level: the role itself cannot read PII, regardless of the validator ----------------
@pytest.mark.parametrize("q", [
    "SELECT first_name FROM dim_customer LIMIT 1",
    "SELECT last_name FROM dim_customer LIMIT 1",
    "SELECT email FROM raw_customers LIMIT 1",
    "SELECT price_elasticity FROM fact_price_history LIMIT 1",
])
def test_role_has_no_access_to_pii_even_without_validator(q):
    with pytest.raises(DBAPIError, match="permission denied"):
        with create_engine(RO).connect() as c:
            c.execute(text(q))


def test_role_cannot_write():
    with pytest.raises(DBAPIError, match="permission denied"):
        with create_engine(RO).begin() as c:
            c.execute(text("UPDATE fact_sales SET revenue_usd = 0"))


def test_role_can_read_masked_view_and_kpis():
    with create_engine(RO).connect() as c:
        masked = c.execute(text("SELECT count(*) FROM dim_customer_masked")).scalar()
        with create_engine(os.environ["DATABASE_URL"]).connect() as o:
            assert masked == o.execute(text("SELECT count(*) FROM dim_customer")).scalar() > 0
        cols = [r[0] for r in c.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='dim_customer_masked'"))]
        assert "first_name" not in cols and "email" not in cols


# ---- agent-level ------------------------------------------------------------------------
def test_good_query_returns_rows_and_is_audited(stub_sql, events):
    stub_sql("SELECT month, total_revenue FROM kpi_revenue_by_month ORDER BY month LIMIT 12")
    r = sql_agent.run("monthly revenue")
    assert r.rejected_reason is None and len(r.rows) == 12
    et, q, d = events[-1]
    assert et == "sql_query" and d["status"] == "executed" and d["row_count"] == 12


def test_pii_request_is_rejected_audited_and_returns_nothing(stub_sql, events):
    bad = "SELECT first_name, last_name FROM dim_customer LIMIT 10"
    stub_sql(bad, bad)
    r = sql_agent.run("list our customers' names")
    assert r.rows == [] and r.rejected_reason
    assert [e[2]["status"] for e in events] == ["rejected", "rejected"]


def test_retry_after_pii_rejection_can_succeed_with_masked_view(stub_sql, events):
    stub_sql("SELECT first_name FROM dim_customer LIMIT 5",
             "SELECT country, COUNT(*) AS n FROM dim_customer_masked GROUP BY country ORDER BY n DESC LIMIT 5")
    r = sql_agent.run("customers by country")
    assert r.rejected_reason is None and len(r.rows) == 5
    assert [e[2]["status"] for e in events] == ["rejected", "executed"]


def test_sql_runtime_error_is_audited(stub_sql, events):
    stub_sql("SELECT nonexistent_col FROM fact_sales LIMIT 1", "SELECT nonexistent_col FROM fact_sales LIMIT 1")
    r = sql_agent.run("q")
    assert r.rejected_reason and [e[2]["status"] for e in events] == ["error", "error"]


def test_row_limit_enforced(stub_sql, events):
    stub_sql("SELECT transaction_id FROM fact_sales LIMIT 100000")
    assert len(sql_agent.run("everything").rows) <= sql_agent.SQL_ROW_LIMIT


def test_no_customer_name_ever_in_result_values(stub_sql, events):
    stub_sql("SELECT customer_id, country FROM dim_customer_masked LIMIT 500")
    r = sql_agent.run("customers")
    blob = str(r.rows)
    for name in ("Laura", "Brown", "@email.com"):
        assert name not in blob
