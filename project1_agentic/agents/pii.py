"""
PII guard for the agentic layer — three independent layers:

  1. Database  : project2_analytics/sql/pii_masking.sql removes the agent role's access to
                 names / e-mails entirely (the real control).
  2. Validator : check_sql() rejects queries that name a PII column or a non-agent table
                 (dim_customer, raw_*) before they reach the database.
  3. Output    : mask_rows() / scrub_text() drop PII columns and redact e-mail addresses and
                 phone numbers that appear in any returned value or in the final answer.
"""
from __future__ import annotations

import re

# Column names that identify a person. `name` is deliberately not here: dim_product.name is a
# product name.
PII_COLUMNS = {
    "first_name", "last_name", "full_name", "customer_name", "email", "email_address",
    "phone", "phone_number", "mobile", "address", "street", "postcode", "zip_code", "ssn", "dob",
    "date_of_birth",
}

# Tables the agent may never read directly (it gets dim_customer_masked instead).
BLOCKED_TABLE_PATTERN = re.compile(r"^(dim_customer|raw_\w+|fact_price_history)$", re.IGNORECASE)

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Separators are mandatory (or a leading +) so a bare 10-digit figure such as an order count or a
# revenue number is not mistaken for a phone number.
_PHONE = re.compile(
    r"(?<![\w.,$])(?:\+\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}(?!\w)"
    r"|(?<![\w.,$])\+\d{10,14}(?!\w)"
)

EMAIL_MASK = "[email removed]"
PHONE_MASK = "[phone removed]"


class PIIViolation(Exception):
    pass


def _strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def check_sql(sql: str) -> None:
    """Raise PIIViolation if the query names a PII column. Table blocking lives in the
    validator's table scan so there is one place that reasons about FROM/JOIN."""
    body = _strip_literals_and_comments(sql).lower()
    for col in PII_COLUMNS:
        if re.search(rf"\b{re.escape(col)}\b", body):
            raise PIIViolation(
                f"Query references personal data column '{col}'. Customer names, e-mail and "
                "contact details are not available to the agent; use customer_id or aggregates."
            )


def scrub_text(value: str) -> str:
    value = _EMAIL.sub(EMAIL_MASK, value)
    return _PHONE.sub(PHONE_MASK, value)


def mask_rows(columns: list[str], rows: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """Drop PII columns and redact PII-looking strings. Returns (columns, rows, dropped)."""
    dropped = [c for c in columns if c.lower() in PII_COLUMNS]
    keep = [c for c in columns if c.lower() not in PII_COLUMNS]
    out = []
    for r in rows:
        clean = {}
        for c in keep:
            v = r.get(c)
            clean[c] = scrub_text(v) if isinstance(v, str) else v
        out.append(clean)
    return keep, out, dropped
