"""
SQL Agent — NL2SQL over the read-only warehouse, with the Diagram 7 safety chain:
SQL Generator -> SQL Validator (reject writes) -> Permission Layer (tables/columns/
row limits/timeouts) -> read-only DB -> Result.

Retries once on a SQL execution error, feeding the real error message back to the LLM
(see evaluation/bibench/runner.py, where this exact approach measurably improved results —
one previously-failing BI-Bench case was fixed purely by giving the retry a real error
instead of a truncated, useless one).
"""
import re
from dataclasses import dataclass

from sqlalchemy import create_engine, text

from config import READONLY_DATABASE_URL, SQL_ROW_LIMIT, SQL_TIMEOUT_MS, call_llm

ALLOWED_TABLES = {
    "dim_customer_safe", "dim_product", "dim_date",
    "fact_sales", "fact_returns", "fact_inventory",
    "fact_supplier_costs", "fact_marketing_spend", "v_price_history_for_agents",
    "kpi_revenue_by_month", "kpi_returns_rate_by_category", "kpi_inventory_turnover",
    "kpi_supplier_risk", "kpi_marketing_roi", "kpi_customer_cohort", "kpi_revenue_by_country_month",
}
# dim_customer itself is deliberately NOT in this list: it carries first_name/last_name.
# The agent reads dim_customer_safe instead (see pii_masking.sql), which the nexus_readonly
# role can actually query — direct SELECT on dim_customer was revoked from that role, so this
# isn't just a prompt-level instruction, the database itself will reject the raw table.

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|CALL)\b", re.IGNORECASE
)

SCHEMA_SUMMARY = """
fact_sales(transaction_id, customer_id, product_id, order_date, quantity, unit_price_usd,
           discount_pct, revenue_usd, cost_usd, profit_usd, channel, payment_method, status, country, category)
   -- status is one of: 'completed', 'refunded', 'cancelled'. "Revenue" in a business
   -- question means completed revenue unless the person explicitly asks for gross/total/
   -- all-status revenue: default to WHERE status = 'completed', matching kpi_revenue_by_month.
fact_returns(return_id, transaction_id, customer_id, product_id, return_date, reason, refund_amount_usd, restocked)
fact_inventory(product_id, category, stock_units, reorder_point, warehouse_location, last_restock_date)
fact_supplier_costs(product_id, supplier_name, reliability_score, lead_time_days, min_order_qty, unit_cost_usd, is_primary)
fact_marketing_spend(year_month, channel, spend_usd, actual_revenue_usd, roas, cac_usd)
dim_customer_safe(customer_id, country, currency, age, gender, registration_date, is_premium, email_verified)
   -- names and email are not available here on purpose; see pii_masking.sql
dim_product(product_id, name, category, brand, unit_price_usd, unit_cost_usd, launch_date)
v_price_history_for_agents(product_id, category, year_month, listed_price_usd, competitor_price_usd, is_promotional, units_sold, revenue_usd, margin_pct)
kpi_revenue_by_month, kpi_returns_rate_by_category, kpi_inventory_turnover, kpi_supplier_risk,
kpi_marketing_roi, kpi_customer_cohort, kpi_revenue_by_country_month  -- pre-aggregated KPI views,
   -- already filtered to completed revenue; prefer these over fact_sales when they cover the question
"""


@dataclass
class SQLResult:
    sql: str
    rows: list
    columns: list
    rejected_reason: str | None = None


class SQLValidationError(Exception):
    pass


def validate_sql(sql: str) -> None:
    if FORBIDDEN_KEYWORDS.search(sql):
        raise SQLValidationError("Query contains a forbidden write/DDL keyword.")
    if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
        raise SQLValidationError("Only SELECT / WITH statements are allowed.")

    # Strip EXTRACT(field FROM source) expressions before scanning for table names —
    # EXTRACT's "FROM" keyword is unrelated to a table reference and would otherwise
    # be misread as one (e.g. "EXTRACT(MONTH FROM r.return_date)" -> false positive on "r").
    sql_for_table_scan = re.sub(r"EXTRACT\s*\([^)]*\)", " ", sql, flags=re.IGNORECASE)

    referenced = set(re.findall(r"\bFROM\s+([a-zA-Z_][\w]*)|\bJOIN\s+([a-zA-Z_][\w]*)", sql_for_table_scan, re.IGNORECASE))
    referenced = {t for pair in referenced for t in pair if t}
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        raise SQLValidationError(f"Query references tables outside the permitted allow-list: {unknown}")

    if not re.search(r"\bLIMIT\s+\d+", sql, re.IGNORECASE):
        raise SQLValidationError("Query must include an explicit LIMIT clause.")


def generate_sql(question: str, retry_error: str | None = None) -> str:
    system = (
        "You write a single read-only PostgreSQL SELECT statement to answer a business "
        "question. Only use the tables/views listed. Always include LIMIT "
        f"{SQL_ROW_LIMIT} or fewer. Return ONLY the SQL, no markdown fences, no commentary."
    )
    prompt = f"Schema:\n{SCHEMA_SUMMARY}\n\nQuestion: {question}\n\nSQL:"
    if retry_error:
        prompt += f"\n\nYour previous SQL failed with this error, fix it:\n{retry_error}"
    sql = call_llm(system, prompt, max_tokens=1000).strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    return sql


def run(question: str) -> SQLResult:
    sql, error = None, None
    for attempt in range(2):  # one retry on SQL error, using the real error message
        sql = generate_sql(question, retry_error=error)
        try:
            validate_sql(sql)
        except SQLValidationError as e:
            error = str(e)
            if attempt == 1:
                return SQLResult(sql=sql, rows=[], columns=[], rejected_reason=error)
            continue

        try:
            engine = create_engine(READONLY_DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text(f"SET statement_timeout = {SQL_TIMEOUT_MS}"))
                result = conn.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchmany(SQL_ROW_LIMIT)]
            return SQLResult(sql=sql, rows=rows, columns=columns)
        except Exception as e:
            # Same fix as evaluation/bibench/runner.py: keep the real DB error message
            # (usually at the END of the exception text, after the echoed query), not a
            # naive head-truncation that throws away the one thing the retry needs to see.
            msg = str(e)
            if sql and sql in msg:
                msg = msg.replace(sql, "<query>")
            error = msg[-400:]
            if attempt == 1:
                return SQLResult(sql=sql, rows=[], columns=[], rejected_reason=error)

    return SQLResult(sql=sql or "", rows=[], columns=[], rejected_reason=error)
