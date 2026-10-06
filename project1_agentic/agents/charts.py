"""
Chart generation: turns the evidence an investigation already holds into renderable chart specs,
so answers are no longer text-only. Deterministic (no LLM): the chart type is chosen from the
shape of the data, and only values that are in the evidence are plotted.

A spec is a plain dict, JSON-safe, renderer-agnostic:
    {"type": "line"|"bar", "title": str, "x_label": str, "y_label": str,
     "x": [...], "series": [{"name": str, "y": [...]}]}
The dashboard turns it into a Plotly figure (see spec_to_plotly_dict / dashboard Ask tab).
"""
from __future__ import annotations

import datetime as dt
import decimal
import re

MAX_CHARTS = 3
MAX_BAR_CATEGORIES = 25
_TIME_NAME = re.compile(r"(month|date|day|week|year|quarter|period)", re.I)
_ID_NAME = re.compile(r"(^id$|_id$|^rank$)", re.I)
_ISO = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float, decimal.Decimal)):
        return float(v)
    return None


def _jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, decimal.Decimal):
        return float(v)
    return v


def _is_time(col: str, values: list) -> bool:
    if any(isinstance(v, (dt.date, dt.datetime)) for v in values):
        return True
    return bool(_TIME_NAME.search(col)) and all(isinstance(v, str) and _ISO.match(v) for v in values if v is not None)


def _pretty(col: str) -> str:
    return col.replace("_", " ").strip().capitalize()


def chart_from_rows(rows: list[dict], title: str | None = None) -> dict | None:
    """Pick line vs bar from the shape of a result set. None if it isn't chartable."""
    if not rows or len(rows) < 2:
        return None
    cols = list(rows[0].keys())
    numeric = [c for c in cols if not _ID_NAME.search(c) and all(_num(r.get(c)) is not None for r in rows)]
    others = [c for c in cols if c not in numeric]
    if not numeric or not others:
        return None
    # category / time axis = first non-numeric column (prefer a time-like one)
    xcol = next((c for c in others if _is_time(c, [r.get(c) for r in rows])), others[0])
    is_time = _is_time(xcol, [r.get(xcol) for r in rows])
    ordered = sorted(rows, key=lambda r: str(_jsonable(r[xcol]))) if is_time else rows
    if not is_time and len(rows) > MAX_BAR_CATEGORIES:
        return None
    ycols = numeric[:3]
    return {
        "type": "line" if is_time else "bar",
        "title": title or f"{_pretty(ycols[0])} by {_pretty(xcol).lower()}",
        "x_label": _pretty(xcol), "y_label": _pretty(ycols[0]) if len(ycols) == 1 else "Value",
        "x": [_jsonable(r[xcol]) for r in ordered],
        "series": [{"name": _pretty(c), "y": [_num(r[c]) for r in ordered]} for c in ycols],
    }


def _drilldown_charts(dd: dict) -> list[dict]:
    out = []
    for dim, items in (dd.get("by_dimension") or {}).items():
        if len(items) >= 2:
            out.append({
                "type": "bar",
                "title": f"Change in {dd.get('metric', 'metric')} by {dim} "
                         f"({dd.get('baseline_period')} to {dd.get('target_period')})",
                "x_label": _pretty(dim), "y_label": "Change",
                "x": [str(i["member"]) for i in items],
                "series": [{"name": "Change", "y": [i["change"] for i in items]}],
            })
    return out[:1]          # the single most useful decomposition; keeps the answer uncluttered


def _ml_charts(ml: dict) -> list[dict]:
    fc = ml.get("demand_forecast") or {}
    by_cat = fc.get("predicted_units_by_category")
    if by_cat and len(by_cat) >= 2:
        return [{"type": "bar", "title": f"Forecast units by category ({fc.get('forecast_month', 'next month')})",
                 "x_label": "Category", "y_label": "Predicted units",
                 "x": list(by_cat.keys()), "series": [{"name": "Predicted units", "y": [float(v) for v in by_cat.values()]}]}]
    return []


def build_charts(investigation) -> list[dict]:
    charts: list[dict] = []
    charts += _drilldown_charts(getattr(investigation, "drilldown", None) or {})
    charts += _ml_charts(investigation.ml_findings or {})
    sql_chart = chart_from_rows(investigation.sql_findings.get("rows") or [])
    if sql_chart:
        charts.append(sql_chart)
    return charts[:MAX_CHARTS]


def spec_to_plotly_dict(spec: dict) -> dict:
    """Plotly figure as a plain dict (also what st.plotly_chart / plotly.js accept)."""
    kind = "scatter" if spec["type"] == "line" else "bar"
    traces = []
    for s in spec["series"]:
        t = {"type": kind, "name": s["name"], "x": spec["x"], "y": s["y"]}
        if kind == "scatter":
            t["mode"] = "lines+markers"
        traces.append(t)
    return {"data": traces, "layout": {"title": {"text": spec["title"]},
                                        "xaxis": {"title": {"text": spec["x_label"]}},
                                        "yaxis": {"title": {"text": spec["y_label"]}},
                                        "showlegend": len(traces) > 1}}
