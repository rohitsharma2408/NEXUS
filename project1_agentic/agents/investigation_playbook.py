"""
Multi-step investigation playbook: "why did <metric> change between two periods?"

Today one SQL query answers one question, so "why did revenue fall in February?" got a single
monthly series and a guess. This playbook runs the real analyst workflow, deterministically and
with no LLM in the loop (so every number is reproducible and verifiable):

  1. quantify  : the metric in both periods and the total change
  2. decompose : for each dimension (country, category, channel) how much of the change each
                 member explains (contribution to total delta, in % of the total change)
  3. context   : did the same calendar transition move the same way in other years?
                 (separates "something happened" from "this is what February always does")

All SQL is templated, passes the same validator as agent-written SQL and runs under the
read-only role.
"""
from __future__ import annotations

import datetime as dt
import re
import statistics
from dataclasses import dataclass

from sqlalchemy import create_engine, text

import sql_agent
from config import READONLY_DATABASE_URL

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_MY = rf"(?P<{{n}}m>{_MONTH_RE})\.?(?:\s*,?\s*(?P<{{n}}y>\d{{{{4}}}}))?"

_CHANGE = re.compile(r"\b(why|fall|fell|drop|dropped|decline|declined|decrease|decreased|increase|"
                     r"increased|rise|rose|grew|growth|change|changed|spike|spiked|dip|dipped|down|up|"
                     r"surge|slump|jump)\b", re.I)

METRICS = {
    "revenue": ("SUM(revenue_usd)", "completed revenue (USD)"),
    "profit": ("SUM(profit_usd)", "completed profit (USD)"),
    "orders": ("COUNT(*)", "completed orders"),
}
DIMENSIONS = ("country", "category", "channel")


@dataclass
class Spec:
    metric: str
    a: tuple[int, int]     # (year, month) baseline
    b: tuple[int, int]     # (year, month) period being explained


def _metric_from(q: str) -> str:
    ql = q.lower()
    if re.search(r"\bprofit|margin\b", ql):
        return "profit"
    if re.search(r"\border(s| volume| count)?\b", ql) and not re.search(r"\brevenue|sales\b", ql):
        return "orders"
    return "revenue"


def _shift(ym: tuple[int, int], k: int) -> tuple[int, int]:
    i = ym[0] * 12 + (ym[1] - 1) + k
    return i // 12, i % 12 + 1


def parse_question(question: str, latest: tuple[int, int]) -> Spec | None:
    """Pull (metric, baseline month, target month) out of the question, or None."""
    q = question.strip()
    if not _CHANGE.search(q):
        return None

    def year_for(month: int, y: str | None) -> int:
        if y:
            return int(y)
        return latest[0] if month <= latest[1] else latest[0] - 1

    pair = re.search(
        _MY.format(n="a") + r"\s*(?:to|vs\.?|versus|compared (?:to|with)|and|->|→)\s*" + _MY.format(n="b"), q, re.I)
    if pair:
        ma, mb = MONTHS[pair["am"].lower()], MONTHS[pair["bm"].lower()]
        ya = year_for(ma, pair["ay"])
        yb = int(pair["by"]) if pair["by"] else (ya if mb >= ma else ya + 1)
        return Spec(_metric_from(q), (ya, ma), (yb, mb))
    if re.search(r"\blast month\b", q, re.I):
        return Spec(_metric_from(q), _shift(latest, -1), latest)
    single = re.search(rf"\b(?:in|during|for|of)\s+" + _MY.format(n="b"), q, re.I)
    if single:
        mb = MONTHS[single["bm"].lower()]
        b = (year_for(mb, single["by"]), mb)
        return Spec(_metric_from(q), _shift(b, -1), b)
    return None


def _month_bounds(ym: tuple[int, int]) -> tuple[dt.date, dt.date]:
    nxt = _shift(ym, 1)
    return dt.date(ym[0], ym[1], 1), dt.date(nxt[0], nxt[1], 1)


def _label(ym: tuple[int, int]) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def _run(engine, sql: str, params: dict | None = None):
    sql_agent.validate_sql(sql)           # templated SQL goes through the same gate as LLM SQL
    with engine.connect() as c:
        res = c.execute(text(sql), params or {})
        return list(res.keys()), [dict(zip(res.keys(), r)) for r in res.fetchall()]


def latest_month(engine) -> tuple[int, int]:
    _, rows = _run(engine, "SELECT MAX(order_date) AS d FROM fact_sales LIMIT 1")
    d = rows[0]["d"]
    return d.year, d.month


def _pct(a: float, b: float) -> float | None:
    return round((b - a) / a * 100, 2) if a else None


def decompose(engine, spec: Spec, top_n: int = 5) -> dict:
    expr, metric_label = METRICS[spec.metric]
    (a0, a1), (b0, b1) = _month_bounds(spec.a), _month_bounds(spec.b)
    params = {"a0": a0, "a1": a1, "b0": b0, "b1": b1}
    case = lambda lo, hi: f"{expr.split('(')[0]}(CASE WHEN order_date >= :{lo} AND order_date < :{hi} THEN {'1' if spec.metric == 'orders' else expr[expr.index('(')+1:-1]} ELSE 0 END)"  # noqa: E731

    _, tot = _run(engine, f"""
        SELECT {case('a0','a1')} AS a, {case('b0','b1')} AS b
        FROM fact_sales WHERE status = 'completed'
          AND ((order_date >= :a0 AND order_date < :a1) OR (order_date >= :b0 AND order_date < :b1)) LIMIT 1""", params)
    ta, tb = float(tot[0]["a"] or 0), float(tot[0]["b"] or 0)
    total_delta = tb - ta
    out = {
        "metric": metric_label,
        "baseline_period": _label(spec.a), "target_period": _label(spec.b),
        "baseline_value": round(ta, 2), "target_value": round(tb, 2),
        "total_change": round(total_delta, 2), "total_change_pct": _pct(ta, tb),
        "by_dimension": {},
    }
    for dim in DIMENSIONS:
        _, rows = _run(engine, f"""
            SELECT {dim} AS member, {case('a0','a1')} AS a, {case('b0','b1')} AS b
            FROM fact_sales WHERE status = 'completed'
              AND ((order_date >= :a0 AND order_date < :a1) OR (order_date >= :b0 AND order_date < :b1))
            GROUP BY {dim} LIMIT 200""", params)
        items = []
        for r in rows:
            va, vb = float(r["a"] or 0), float(r["b"] or 0)
            d = vb - va
            items.append({"member": r["member"], "baseline": round(va, 2), "target": round(vb, 2),
                          "change": round(d, 2), "change_pct": _pct(va, vb),
                          "share_of_total_change_pct": round(d / total_delta * 100, 1) if total_delta else None})
        items.sort(key=lambda x: abs(x["change"]), reverse=True)
        out["by_dimension"][dim] = items[:top_n]
    return out


def seasonal_context(engine, spec: Spec) -> dict:
    """Same calendar transition in the other years of data."""
    expr, _ = METRICS[spec.metric]
    _, rows = _run(engine, f"""
        SELECT EXTRACT(YEAR FROM order_date)::int AS y, EXTRACT(MONTH FROM order_date)::int AS m, {expr} AS v
        FROM fact_sales WHERE status = 'completed' GROUP BY 1, 2 LIMIT 500""")
    val = {(r["y"], r["m"]): float(r["v"]) for r in rows}
    this = _pct(val.get(spec.a, 0), val.get(spec.b, 0))
    others = {}
    for (y, m), v in val.items():
        if m == spec.a[1] and (y, m) != spec.a:
            nxt = (y + (spec.b[0] - spec.a[0]), spec.b[1])
            if nxt in val and v:
                others[_label((y, m)) + "->" + _label(nxt)] = _pct(v, val[nxt])
    typical = round(statistics.median(others.values()), 2) if others else None
    consistent = None
    if this is not None and typical is not None:
        consistent = (this * typical > 0) and abs(this - typical) <= 10
    return {"this_transition_pct": this, "same_transition_other_years_pct": others,
            "typical_pct_change": typical, "consistent_with_seasonality": consistent}


def investigate_change(question: str, engine=None) -> dict | None:
    """Returns a structured drill-down, or None if the question isn't a period-change question."""
    engine = engine or create_engine(READONLY_DATABASE_URL)
    spec = parse_question(question, latest_month(engine))
    if spec is None or spec.a == spec.b:
        return None
    result = decompose(engine, spec)
    result["seasonality"] = seasonal_context(engine, spec)
    top = []
    for dim, items in result["by_dimension"].items():
        if items and items[0]["share_of_total_change_pct"] is not None:
            top.append(f"{dim}: {items[0]['member']} ({items[0]['share_of_total_change_pct']}% of the change)")
    result["top_drivers_summary"] = top
    return result
