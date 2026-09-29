"""Smoke tests for the SQL Agent's safety validator (no DB/LLM calls needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project1_agentic"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project1_agentic" / "agents"))

from sql_agent import validate_sql, SQLValidationError  # noqa: E402


def test_rejects_drop():
    try:
        validate_sql("DROP TABLE fact_sales")
        assert False, "should have raised"
    except SQLValidationError:
        pass


def test_rejects_missing_limit():
    try:
        validate_sql("SELECT * FROM fact_sales")
        assert False, "should have raised"
    except SQLValidationError:
        pass


def test_rejects_unknown_table():
    try:
        validate_sql("SELECT * FROM pg_shadow LIMIT 10")
        assert False, "should have raised"
    except SQLValidationError:
        pass


def test_accepts_valid_select():
    validate_sql("SELECT category, SUM(revenue_usd) FROM fact_sales GROUP BY category LIMIT 50")
