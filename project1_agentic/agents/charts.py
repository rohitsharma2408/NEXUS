"""
Chart generation: turns the evidence an investigation already holds into renderable chart specs,
so answers are no longer text-only. Deterministic (no LLM): the chart type is chosen from the
shape of the data, and only values that are in the evidence are plotted.

A spec is a plain dict, JSON-safe, renderer-agnostic:
    {"type": "line"|"bar"|"pie", "title": str, "x_label": str, "y_label": str,
     "x": [...], "series": [{"name": str, "y": [...]}]}
The dashboard turns it into a Plotly figure (see spec_to_plotly_dict / dashboard Ask tab).
"""
from __future__ import annotations

import datetime as dt
import decimal
import re

MAX_CHARTS = 3
MAX_BAR_CATEGORIES = 25
MAX_PIE_SLICES = 12          # larger pies are unreadable: the smallest slices fold into "Other"
MAX_SPLIT = 8                # a second category becomes one series per value only up to this many
_PIE_WORDS = re.compile(r"\b(pie|donut|doughnut)\b", re.I)
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


def _slice_label(v) -> str:
    v = _jsonable(v)
    if isinstance(v, str) and _ISO.match(v):
        try:
            return dt.datetime.strptime(v[:7], "%Y-%m").strftime("%b %Y")
        except ValueError:
            return v
    return str(v)


def pie_from_rows(rows: list[dict], title: str | None = None) -> dict | None:
    """One slice per category (summed if a category repeats). Negative/zero values can't be sliced."""
    if not rows or len(rows) < 2:
        return None
    cols = list(rows[0].keys())
    numeric = [c for c in cols if not _ID_NAME.search(c) and all(_num(r.get(c)) is not None for r in rows)]
    others = [c for c in cols if c not in numeric]
    if not numeric or not others:
        return None
    xcol = next((c for c in others if _is_time(c, [r.get(c) for r in rows])), others[0])
    ycol = numeric[0]
    totals: dict[str, float] = {}
    for r in rows:
        v = _num(r[ycol])
        if v is not None and v > 0:
            key = _slice_label(r[xcol])
            totals[key] = totals.get(key, 0.0) + v
    if len(totals) < 2:
        return None
    items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) > MAX_PIE_SLICES:
        keep, rest = items[:MAX_PIE_SLICES - 1], items[MAX_PIE_SLICES - 1:]
        items = keep + [(f"Other ({len(rest)})", sum(v for _, v in rest))]
    return {"type": "pie", "title": title or f"{_pretty(ycol)} by {_pretty(xcol).lower()}",
            "x_label": _pretty(xcol), "y_label": _pretty(ycol),
            "x": [k for k, _ in items], "series": [{"name": _pretty(ycol), "y": [v for _, v in items]}]}


def chart_from_rows(rows: list[dict], title: str | None = None, prefer_pie: bool = False) -> dict | None:
    """Pick line vs bar (or pie when asked) from the shape of a result set. None if not chartable.
    Several rows per x value (e.g. one per channel per month) are split into one series per
    category, or summed when there is no usable category, so lines never zigzag."""
    if prefer_pie:
        pie = pie_from_rows(rows, title)
        if pie:
            return pie
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
    ycols = numeric[:3]
    xkeys = [str(_jsonable(r[xcol])) for r in rows]
    if len(set(xkeys)) < len(xkeys):                      # repeated x: split or aggregate
        split = next((c for c in others if c != xcol and 2 <= len({str(r[c]) for r in rows}) <= MAX_SPLIT), None)
        order = sorted(set(xkeys)) if is_time else list(dict.fromkeys(xkeys))
        if split:
            y0 = ycols[0]
            groups: dict[str, dict[str, float]] = {}
            for r in rows:
                g = groups.setdefault(str(r[split]), {})
                g[str(_jsonable(r[xcol]))] = g.get(str(_jsonable(r[xcol])), 0.0) + (_num(r[y0]) or 0.0)
            return {"type": "line" if is_time else "bar",
                    "title": title or f"{_pretty(y0)} by {_pretty(xcol).lower()} and {_pretty(split).lower()}",
                    "x_label": _pretty(xcol), "y_label": _pretty(y0), "x": order,
                    "series": [{"name": name, "y": [g.get(x) for x in order]} for name, g in groups.items()]}
        sums = {c: dict.fromkeys(order, 0.0) for c in ycols}
        for r, k in zip(rows, xkeys):
            for c in ycols:
                sums[c][k] += _num(r[c]) or 0.0
        return {"type": "line" if is_time else "bar",
                "title": title or f"{_pretty(ycols[0])} by {_pretty(xcol).lower()}",
                "x_label": _pretty(xcol), "y_label": _pretty(ycols[0]) if len(ycols) == 1 else "Value",
                "x": order, "series": [{"name": _pretty(c), "y": [sums[c][x] for x in order]} for c in ycols]}
    ordered = sorted(rows, key=lambda r: str(_jsonable(r[xcol]))) if is_time else rows
    if not is_time and len(rows) > MAX_BAR_CATEGORIES:
        return None
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
    rows = investigation.sql_findings.get("rows") or []
    wants_pie = bool(_PIE_WORDS.search(getattr(investigation, "question", "") or ""))
    if wants_pie:                       # the chart they asked for goes first, the regular one beside it
        pie = pie_from_rows(rows)
        if pie:
            charts.append(pie)
    charts += _drilldown_charts(getattr(investigation, "drilldown", None) or {})
    charts += _ml_charts(investigation.ml_findings or {})
    sql_chart = chart_from_rows(rows)
    if sql_chart:
        charts.append(sql_chart)
    return charts[:MAX_CHARTS]


def spec_to_plotly_dict(spec: dict) -> dict:
    """Plotly figure as a plain dict (also what st.plotly_chart / plotly.js accept)."""
    if spec["type"] == "pie":
        return {"data": [{"type": "pie", "labels": spec["x"], "values": spec["series"][0]["y"], "hole": 0.45}],
                "layout": {"title": {"text": spec["title"]}, "showlegend": True}}
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
