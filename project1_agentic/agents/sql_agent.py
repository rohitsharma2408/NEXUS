"""
SQL Agent — NL2SQL over the read-only warehouse, with the Diagram 7 safety chain:
SQL Generator -> SQL Validator (reject writes) -> Permission Layer (tables/columns/
row limits/timeouts) -> read-only DB -> Result.
"""
import re
from dataclasses import dataclass

from sqlalchemy import create_engine, text

from config import READONLY_DATABASE_URL, SQL_ROW_LIMIT, SQL_TIMEOUT_MS, call_llm

ALLOWED_TABLES = {
    "dim_customer", "dim_product", "dim_date",
    "fact_sales", "fact_returns", "fact_inventory",
    "fact_supplier_costs", "fact_marketing_spend", "v_price_history_for_agents",
    "kpi_revenue_by_month", "kpi_returns_rate_by_category", "kpi_inventory_turnover",
    "kpi_supplier_risk", "kpi_marketing_roi", "kpi_customer_cohort", "kpi_revenue_by_country_month",
}

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|CALL)\b", re.IGNORECASE
)

SCHEMA_SUMMARY = """
fact_sales(transaction_id, customer_id, product_id, order_date, quantity, unit_price_usd,
           discount_pct, revenue_usd, cost_usd, profit_usd, channel, payment_method, status, country, category)
fact_returns(return_id, transaction_id, customer_id, product_id, return_date, reason, refund_amount_usd, restocked)
fact_inventory(product_id, category, stock_units, reorder_point, warehouse_location, last_restock_date)
fact_supplier_costs(product_id, supplier_name, reliability_score, lead_time_days, min_order_qty, unit_cost_usd, is_primary)
fact_marketing_spend(year_month, channel, spend_usd, actual_revenue_usd, roas, cac_usd)
dim_customer(customer_id, country, age, gender, registration_date, is_premium)
dim_product(product_id, name, category, brand, unit_price_usd, unit_cost_usd, launch_date)
v_price_history_for_agents(product_id, category, year_month, listed_price_usd, competitor_price_usd, is_promotional, units_sold, revenue_usd, margin_pct)
kpi_revenue_by_month, kpi_returns_rate_by_category, kpi_inventory_turnover, kpi_supplier_risk,
kpi_marketing_roi, kpi_customer_cohort, kpi_revenue_by_country_month  -- pre-aggregated KPI views
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


def generate_sql(question: str) -> str:
    system = (
        "You write a single read-only PostgreSQL SELECT statement to answer a business "
        "question. Only use the tables/views listed. Always include LIMIT "
        f"{SQL_ROW_LIMIT} or fewer. Return ONLY the SQL, no markdown fences, no commentary."
    )
    prompt = f"Schema:\n{SCHEMA_SUMMARY}\n\nQuestion: {question}\n\nSQL:"
    sql = call_llm(system, prompt, max_tokens=1000).strip()
    sql = re.sub(r"^```sql|```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    return sql


def run(question: str) -> SQLResult:
    sql = generate_sql(question)
    try:
        validate_sql(sql)
    except SQLValidationError as e:
        return SQLResult(sql=sql, rows=[], columns=[], rejected_reason=str(e))

    engine = create_engine(READONLY_DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text(f"SET statement_timeout = {SQL_TIMEOUT_MS}"))
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchmany(SQL_ROW_LIMIT)]
    return SQLResult(sql=sql, rows=rows, columns=columns)
