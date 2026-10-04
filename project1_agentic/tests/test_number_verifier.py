"""Run: docker compose exec api python project1_agentic/tests/test_number_verifier.py"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agents"))
from number_verifier import verify_answer  # noqa: E402


def inv(question="Which month had the highest revenue?", rows=None, ml=None, chunks=None):
    return SimpleNamespace(
        question=question,
        sql_findings={"rows": rows or []},
        ml_findings=ml or {},
        rag_findings={"chunks": chunks or []},
    )


ROWS = [
    {"month": "2023-12-01T00:00:00Z", "total_revenue": 2115457.409999999, "order_count": 3984, "aov": "530.99"},
    {"month": "2022-12-01T00:00:00Z", "total_revenue": 2000165.72, "order_count": 3976, "aov": "503.06"},
]


def test_exact_and_rounded_supported():
    r = verify_answer("December 2023 revenue was $2,115,457.41 (about $2.1M) from 3,984 orders.", inv(rows=ROWS))
    assert r["unsupported"] == [], r
    assert len(r["supported"]) == 3, r


def test_fabricated_number_flagged():
    r = verify_answer("Revenue was $2,900,000 across 3,984 orders.", inv(rows=ROWS))
    assert r["unsupported"] == ["$2,900,000"], r


def test_derived_difference_and_pct_change():
    r = verify_answer("That is $115,291.69 more than 2022, a 5.8% increase.", inv(rows=ROWS))
    assert r["unsupported"] == [], r
    assert "$115,291.69" in r["derived"], r


def test_column_total_is_derived():
    r = verify_answer("Combined revenue was $4,115,623.13.", inv(rows=ROWS))
    assert r["unsupported"] == [], r


def test_years_dates_and_small_counts_ignored():
    r = verify_answer("In December 2023, the top 5 months were all Q4.", inv(rows=ROWS))
    assert r["checked"] == 0, r


def test_ids_not_treated_as_numbers():
    r = verify_answer("PROD0055 leads with 123.88 units.", inv(ml={"p": [{"id": "PROD0055", "u": 123.88079}]}))
    assert r["unsupported"] == [] and r["checked"] == 1, r


def test_percent_matches_fraction_and_rag_text():
    r = verify_answer("The return rate was 15.2%, above the 10% investigation threshold.",
                      inv(rows=[{"rate": 0.152}], chunks=[{"content": "exceeds a 10% return rate"}]))
    assert r["unsupported"] == [], r


def test_number_from_question_allowed():
    r = verify_answer("Over the next 24 months, the data shows no forecast.", inv(question="Forecast 24 months"))
    assert r["unsupported"] == [], r


def test_no_evidence_means_everything_unsupported():
    r = verify_answer("Revenue was $1,234,567.", inv())
    assert r["unsupported"] == ["$1,234,567"], r


def test_dates_are_not_figures():
    r = verify_answer("Between January 1 and December 31, 2022, 1 premium customer registered; "
                      "on 31 December 2022 the count was 1,000 less than expected.", inv(rows=[{"count": 1}]))
    assert r["unsupported"] == ["1,000"], r


def test_plain_count_near_a_date_still_checked():
    r = verify_answer("31 customers registered.", inv(rows=[{"count": 1}]))
    assert r["unsupported"] == ["31"], r


def test_small_count_confirmed_when_it_matches():
    r = verify_answer("Exactly 1 premium customer registered between January 1 and December 31, 2022.",
                      inv(question="How many premium customers registered in 2022?", rows=[{"count": 1}]))
    assert r["supported"] == ["1"] and r["unsupported"] == [] and r["checked"] == 1, r


def test_top_n_with_no_match_is_ignored_not_flagged():
    r = verify_answer("Here are the top 5 months.", inv(rows=[{"count": 1}]))
    assert r["checked"] == 0 and r["unsupported"] == [], r


def test_date_digits_in_evidence_do_not_support_small_numbers():
    r = verify_answer("There were 12 months.", inv(rows=[{"month": "2023-12-01T00:00:00Z"}]))
    assert r["supported"] == [], r


def test_known_limit_wrong_small_count_is_not_flagged():
    # Documented weakness: a wrong small count (5 instead of 1) is ignored, not flagged,
    # because small integers are only confirmed positively. Larger figures are still checked.
    r = verify_answer("Exactly 5 premium customers registered.", inv(rows=[{"count": 1}]))
    assert r["unsupported"] == [], r


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"\n{len(tests)} passed")
