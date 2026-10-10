"""SUM() over a NUMERIC column reaches the verifier as decimal.Decimal; it must count as evidence."""
import decimal
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
from number_verifier import verify_answer  # noqa: E402

ROWS = [{"month": "2024-12-01", "total_units_sold": decimal.Decimal("9077")},
        {"month": "2024-11-01", "total_units_sold": decimal.Decimal("7858")}]


def _inv():
    return SimpleNamespace(question="Forecast next month's total units sold.",
                           sql_findings={"rows": ROWS}, ml_findings={}, rag_findings={"chunks": []})


def test_decimal_evidence_is_recognised():
    r = verify_answer("December 2024 had 9,077 units and November 2024 had 7,858 units.", _inv())
    assert r["unsupported"] == [], r


def test_invented_number_is_still_flagged():
    r = verify_answer("December 2024 had 9,999 units.", _inv())
    assert "9,999" in r["unsupported"], r
