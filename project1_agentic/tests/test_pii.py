"""PII guard + hardened SQL validator. Pure offline tests (no DB, no LLM)."""
import pytest

import pii
from sql_agent import SQLValidationError, referenced_tables, validate_sql


def rejects(sql, fragment=None):
    with pytest.raises(SQLValidationError) as e:
        validate_sql(sql)
    if fragment:
        assert fragment.lower() in str(e.value).lower(), str(e.value)


# ---- PII at the validator ---------------------------------------------------------------
def test_blocks_customer_names_via_masked_view_column():
    rejects("SELECT first_name, last_name FROM dim_customer_masked LIMIT 5", "personal data")

def test_blocks_raw_customer_table():
    rejects("SELECT customer_id FROM dim_customer LIMIT 5", "not available")
    rejects("SELECT * FROM raw_customers LIMIT 5", "not available")

def test_blocks_email_column():
    rejects("SELECT email FROM dim_customer_masked LIMIT 5", "personal data")

def test_blocks_schema_qualified():
    rejects("SELECT * FROM public.dim_customer LIMIT 5", "not available")

def test_blocks_comma_join_bypass():
    rejects("SELECT s.revenue_usd FROM fact_sales s, dim_customer c LIMIT 5", "not available")

def test_blocks_subquery_bypass():
    rejects("SELECT * FROM fact_sales WHERE customer_id IN (SELECT customer_id FROM raw_customers) LIMIT 5")

def test_blocks_pii_hidden_by_comment_trick():
    rejects("SELECT customer_id, /* harmless */ first_name FROM dim_customer_masked LIMIT 5")

def test_blocks_elasticity_table():
    rejects("SELECT price_elasticity FROM fact_price_history LIMIT 5", "not available")

def test_literal_mentioning_pii_word_is_fine():
    validate_sql("SELECT COUNT(*) FROM fact_sales WHERE channel = 'email' LIMIT 5")

def test_masked_view_accepted():
    validate_sql("SELECT country, COUNT(*) FROM dim_customer_masked GROUP BY country LIMIT 50")


# ---- validator hardening ------------------------------------------------------------------
def test_cte_names_are_not_unknown_tables():
    validate_sql("WITH m AS (SELECT month, total_revenue FROM kpi_revenue_by_month) "
                 "SELECT * FROM m ORDER BY month LIMIT 10")

def test_extract_from_is_not_a_table():
    validate_sql("SELECT EXTRACT(MONTH FROM order_date) AS m, SUM(revenue_usd) FROM fact_sales "
                 "GROUP BY 1 LIMIT 12")

def test_extract_with_nested_parens():
    validate_sql("SELECT EXTRACT(YEAR FROM DATE_TRUNC('month', order_date)) FROM fact_sales LIMIT 5")

def test_joins_and_aliases_accepted():
    validate_sql("SELECT p.category, SUM(s.revenue_usd) FROM fact_sales AS s "
                 "JOIN dim_product p ON s.product_id = p.product_id GROUP BY 1 LIMIT 20")
    assert referenced_tables("SELECT 1 FROM fact_sales s, dim_product p LIMIT 1") == {"fact_sales", "dim_product"}

def test_rejects_multiple_statements():
    rejects("SELECT 1 FROM fact_sales LIMIT 1; SELECT 2 FROM fact_sales LIMIT 1", "single statement")

def test_rejects_select_into():
    rejects("SELECT * INTO scratch FROM fact_sales LIMIT 5", "forbidden")

def test_rejects_dangerous_functions():
    rejects("SELECT pg_read_file('/etc/passwd') LIMIT 1", "forbidden")
    rejects("SELECT pg_sleep(30) LIMIT 1", "forbidden")

def test_rejects_system_catalogs():
    rejects("SELECT * FROM pg_shadow LIMIT 5")
    rejects("SELECT * FROM information_schema.tables LIMIT 5")

def test_write_keyword_in_comment_or_literal_does_not_false_positive():
    validate_sql("SELECT COUNT(*) FROM fact_sales WHERE status = 'update' LIMIT 1 -- drop me")

def test_write_keyword_for_real_still_rejected():
    rejects("SELECT 1 FROM fact_sales LIMIT 1 /* x */ ; DROP TABLE fact_sales")


# ---- output masking -----------------------------------------------------------------------
def test_mask_rows_drops_pii_columns_and_redacts_values():
    cols = ["customer_id", "first_name", "note"]
    rows = [{"customer_id": "C1", "first_name": "Laura", "note": "mail laura@x.com or call 415-555-0199"}]
    c, r, dropped = pii.mask_rows(cols, rows)
    assert c == ["customer_id", "note"] and dropped == ["first_name"]
    assert "laura@x.com" not in r[0]["note"] and "415-555-0199" not in r[0]["note"]
    assert pii.EMAIL_MASK in r[0]["note"] and pii.PHONE_MASK in r[0]["note"]

def test_scrub_text_leaves_normal_numbers_alone():
    t = "Revenue was $2,115,457.41 from 3,984 orders in 2023-12 (AOV 530.99)."
    assert pii.scrub_text(t) == t

def test_scrub_text_removes_email():
    assert "bob@corp.io" not in pii.scrub_text("Contact bob@corp.io about it")

def test_bare_large_numbers_are_not_phones():
    t = "Total units 1234567890 and revenue 9876543210 across the period."
    assert pii.scrub_text(t) == t

def test_international_phone_removed():
    assert "+447911123456" not in pii.scrub_text("call +447911123456 now")
