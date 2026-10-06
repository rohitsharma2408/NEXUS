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

import time

import audit
import nl2sql_core
import pii
from config import READONLY_DATABASE_URL, SQL_ROW_LIMIT, SQL_TIMEOUT_MS, call_llm

ALLOWED_TABLES = {
    "dim_customer_masked", "dim_product", "dim_date",
    "fact_sales", "fact_returns", "fact_inventory",
    "fact_supplier_costs", "fact_marketing_spend", "v_price_history_for_agents",
    "kpi_revenue_by_month", "kpi_returns_rate_by_category", "kpi_inventory_turnover",
    "kpi_supplier_risk", "kpi_marketing_roi", "kpi_customer_cohort", "kpi_revenue_by_country_month",
}

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE|CREATE|EXEC|EXECUTE|CALL|COPY|MERGE|"
    r"VACUUM|REFRESH|LISTEN|NOTIFY|PREPARE|SET|RESET|LOAD|INTO|DO)\b", re.IGNORECASE
)
FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(pg_read_file|pg_read_binary_file|pg_ls_dir|pg_stat_file|lo_import|lo_export|dblink\w*|"
    r"set_config|pg_sleep|pg_terminate_backend|pg_cancel_backend|current_setting|query_to_xml)\s*\(",
    re.IGNORECASE,
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
dim_customer_masked(customer_id, country, currency, age, gender, registration_date, is_premium, email_verified)
   -- customer names, e-mail and contact details are NOT available; use customer_id or aggregates
dim_product(product_id, name, category, brand, unit_price_usd, unit_cost_usd, launch_date)
v_price_history_for_agents(product_id, category, year_month, listed_price_usd, base_price_usd, competitor_price_usd, price_index, is_promotional, units_sold, revenue_usd, margin_pct)
-- pre-aggregated KPI views, already filtered to completed revenue; prefer these over fact_sales when they cover the question:
kpi_revenue_by_month(month, total_revenue, total_profit, order_count, aov)
kpi_revenue_by_country_month(country, month, revenue, profit, orders)
kpi_returns_rate_by_category(category, return_count, total_sales, return_rate_pct)
kpi_inventory_turnover(product_id, category, stock_units, reorder_point, units_sold_90d, months_of_stock, at_reorder_risk)
kpi_supplier_risk(product_id, supplier_name, reliability_score, lead_time_days, min_order_qty, unit_cost_usd, stock_units, reorder_point, supplier_risk_level)
kpi_marketing_roi(year_month, channel, spend_usd, actual_revenue_usd, roas, cac_usd, revenue_per_spend)
kpi_customer_cohort(cohort_month, is_premium, customers, orders, revenue)
-- RULES: SQL only reports historical data. Never build a forecast or projection in SQL
-- (no shifting dates forward, no averaging past months as a "forecast"). For forecast or
-- prediction questions, return the relevant historical series only (e.g. monthly revenue
-- from kpi_revenue_by_month, ordered by month) and leave the prediction to the ML model.
"""


@dataclass
class SQLResult:
    sql: str
    rows: list
    columns: list
    rejected_reason: str | None = None


class SQLValidationError(Exception):
    pass


_RESERVED_AFTER_TABLE = {
    "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "UNION", "INTERSECT", "EXCEPT", "JOIN", "INNER",
    "LEFT", "RIGHT", "FULL", "CROSS", "ON", "USING", "WINDOW", "OFFSET", "FETCH", "NATURAL", "LATERAL",
}
_FROM_INSIDE_FUNCS = {"EXTRACT", "TRIM", "SUBSTRING", "OVERLAY", "POSITION"}


def _clean(sql: str) -> str:
    """Remove comments and string literals so they can neither hide nor fake a keyword."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def referenced_tables(sql: str) -> set[str]:
    """Every table/view named after FROM or JOIN, including comma joins, schema-qualified names
    and nested subqueries. CTE names defined in the query are excluded."""
    s = _clean(sql)
    ctes = {m.lower() for m in re.findall(
        r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)([a-zA-Z_]\w*)\s+AS\s*\(", s, flags=re.I)}
    toks = re.findall(r"[A-Za-z_][\w.]*|\d+(?:\.\d+)?|,|\(|\)|[^\s\w,()]", s)
    found: set[str] = set()
    stack: list[bool] = []          # True = inside EXTRACT(...)-style parens where FROM isn't a table
    i, expect = 0, False
    while i < len(toks):
        t = toks[i]
        u = t.upper()
        if t == "(":
            prev = toks[i - 1].upper() if i else ""
            stack.append(prev in _FROM_INSIDE_FUNCS)
            if expect:
                expect = False       # subquery: its own FROM will re-arm expect
            i += 1
            continue
        if t == ")":
            if stack:
                stack.pop()
            i += 1
            continue
        if u in ("FROM", "JOIN") and not (stack and stack[-1]):
            expect = True
            i += 1
            continue
        if expect and re.match(r"[A-Za-z_]", t):
            name = t.split(".")[-1].lower()
            if name not in ctes:
                found.add(name)
            i += 1
            if i < len(toks) and toks[i].upper() == "AS":
                i += 1
            if i < len(toks) and re.match(r"[A-Za-z_]", toks[i]) and toks[i].upper() not in _RESERVED_AFTER_TABLE:
                i += 1               # alias
            if i < len(toks) and toks[i] == ",":
                i += 1               # comma join: next table follows
                continue
            expect = False
            continue
        i += 1
    return found


def validate_sql(sql: str) -> None:
    cleaned = _clean(sql)
    body = cleaned.strip().rstrip(";").strip()
    if ";" in body:
        raise SQLValidationError("Only a single statement is allowed.")
    if FORBIDDEN_KEYWORDS.search(cleaned):
        raise SQLValidationError("Query contains a forbidden write/DDL keyword.")
    if FORBIDDEN_FUNCTIONS.search(cleaned):
        raise SQLValidationError("Query calls a forbidden system function.")
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise SQLValidationError("Only SELECT / WITH statements are allowed.")

    for t in referenced_tables(sql):
        if pii.BLOCKED_TABLE_PATTERN.match(t):
            raise SQLValidationError(
                f"Table '{t}' is not available to the agent (personal or validation-only data). "
                "Use dim_customer_masked / v_price_history_for_agents instead."
            )
    unknown = referenced_tables(sql) - ALLOWED_TABLES
    if unknown:
        raise SQLValidationError(f"Query references tables outside the permitted allow-list: {unknown}")

    try:
        pii.check_sql(sql)
    except pii.PIIViolation as e:
        raise SQLValidationError(str(e))

    if not re.search(r"\bLIMIT\s+\d+", cleaned, re.IGNORECASE):
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
    return nl2sql_core.strip_fences(call_llm(system, prompt, max_tokens=1000))


def run(question: str) -> SQLResult:
    """generate -> validate -> read-only execute -> PII-mask, with one retry fed the real error.
    The loop itself lives in nl2sql_core (shared with the BI-Bench runner)."""
    engine = create_engine(READONLY_DATABASE_URL)
    state = {"t0": time.time(), "dropped": []}

    def execute(sql: str):
        state["t0"] = time.time()
        with engine.connect() as conn:
            conn.execute(text(f"SET statement_timeout = {SQL_TIMEOUT_MS}"))
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, row)) for row in result.fetchmany(SQL_ROW_LIMIT)]
        columns, rows, state["dropped"] = pii.mask_rows(columns, rows)   # defence in depth
        return columns, rows

    def on_attempt(a, n):
        extra = {}
        if a.status == "executed":
            extra = {"row_count": len(state["_rows"]), "columns": state["_cols"],
                     "pii_columns_dropped": state["dropped"]}
        audit.log_event("sql_query", question, status=a.status, reason=a.reason, sql=a.sql, attempt=n,
                        duration_ms=int((time.time() - state["t0"]) * 1000), **extra)

    def execute_and_remember(sql):
        cols, rows = execute(sql)
        state["_cols"], state["_rows"] = cols, rows
        return cols, rows

    def validate(sql):
        try:
            validate_sql(sql)
        except SQLValidationError as e:
            raise nl2sql_core.Rejected(str(e))

    out = nl2sql_core.run_with_retry(
        generate=lambda err: generate_sql(question, retry_error=err),
        execute=execute_and_remember, validate=validate, max_attempts=2, on_attempt=on_attempt)
    if out.error is not None:
        return SQLResult(sql=out.sql, rows=[], columns=[], rejected_reason=out.error)
    cols, rows = out.result
    return SQLResult(sql=out.sql, rows=rows, columns=cols)
