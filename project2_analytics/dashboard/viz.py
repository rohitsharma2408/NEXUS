"""Plotly styling + figure builders shared by the NEXUS dashboard (dark / gold theme)."""
from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

GOLD = "#D4AF37"
PALETTE = [GOLD, "#5B8DEF", "#E86F8C", "#4FD1A5", "#B794F4", "#F6AD55", "#63B3ED", "#FC8181"]
GRID = "rgba(255,255,255,0.07)"
UP, DOWN = "#4FD1A5", "#FC8181"
LEVEL_COLORS = {"LOW": "#4FD1A5", "MEDIUM": "#F6AD55", "HIGH": "#FC8181"}
# scrollZoom: mouse-wheel zoom; box-zoom/pan/reset come with the modebar.
PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True,
               "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
KINDS = ["line", "bar", "area", "scatter"]
_DATE_NAME = re.compile(r"^(month|year_month|order_date|cohort_month|date|day|week)$", re.I)
_ISO = re.compile(r"^\d{4}-\d{2}(-\d{2})?")


def register_template() -> None:
    pio.templates["nexus"] = go.layout.Template(layout=go.Layout(
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8E6E3", family="Inter, Segoe UI, Roboto, sans-serif", size=13),
        title=dict(font=dict(size=16), x=0.0, xanchor="left"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title=None),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    title=dict(text="")),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified", hoverlabel=dict(bgcolor="#141B2F"),
    ))
    pio.templates.default = "nexus"


def is_temporal(df: pd.DataFrame, col: str) -> bool:
    """True when `col` is a date axis (real datetimes, or ISO strings in a date-like column)."""
    if col not in df.columns:
        return False
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object and _DATE_NAME.match(col):
        vals = [v for v in s.dropna().head(20)]
        return bool(vals) and all(isinstance(v, str) and _ISO.match(v) or hasattr(v, "year") for v in vals)
    return False


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(template="nexus", height=height)
    return fig


def spec_to_fig(spec: dict, height: int = 360) -> go.Figure:
    """Chart spec produced by the analyst API (see agents/charts.py) -> interactive Plotly figure."""
    fig = go.Figure()
    if spec.get("type") == "pie":
        s0 = spec["series"][0]
        fig.add_trace(go.Pie(labels=spec["x"], values=s0["y"], hole=0.45, sort=False, textinfo="label+percent",
                             hovertemplate="%{label}<br>%{value:,.2f} (%{percent})<extra></extra>"))
        fig.update_layout(title=dict(text=spec.get("title", "")), showlegend=True, hovermode="closest")
        return style(fig, height)
    is_line = spec.get("type") == "line"
    for s in spec.get("series", []):
        ys = s.get("y", [])
        if is_line:
            fig.add_trace(go.Scatter(x=spec["x"], y=ys, name=s["name"], mode="lines+markers"))
        else:
            kwargs = {}
            if len(spec["series"]) == 1 and s["name"].lower() == "change":
                kwargs["marker_color"] = [UP if (v or 0) >= 0 else DOWN for v in ys]
            fig.add_trace(go.Bar(x=spec["x"], y=ys, name=s["name"], **kwargs))
    fig.update_layout(title=dict(text=spec.get("title", "")), barmode="group",
                      showlegend=len(spec.get("series", [])) > 1)
    fig.update_xaxes(title=dict(text=spec.get("x_label", "")))
    fig.update_yaxes(title=dict(text=spec.get("y_label", "")))
    return style(fig, height)


def df_fig(df: pd.DataFrame, kind: str, x: str, ys: list[str], color: str | None = None,
           title: str = "", height: int = 360) -> go.Figure:
    """Interactive figure from a dataframe. One y + optional colour split, or several y columns."""
    if not ys:
        raise ValueError("pick at least one value column")
    d = df
    if is_temporal(d, x) and not pd.api.types.is_datetime64_any_dtype(d[x]):
        d = d.assign(**{x: pd.to_datetime(d[x], errors="coerce")})   # 'YYYY-MM' strings -> real dates
    if kind in ("line", "area"):
        d = d.sort_values(x)
    args: dict = dict(data_frame=d, x=x, title=title, template="nexus")
    if color and len(ys) == 1:
        args.update(y=ys[0], color=color)
    else:
        args.update(y=ys)
    if kind == "line":
        fig = px.line(markers=True, **args)
    elif kind == "bar":
        fig = px.bar(barmode="group", **args)
    elif kind == "area":
        fig = px.area(**args)
    elif kind == "scatter":
        fig = px.scatter(**args)
    else:
        raise ValueError(f"unknown chart kind: {kind}")
    if len(ys) > 1 and not color:
        fig.for_each_trace(lambda t: t.update(name=t.name.replace("_", " ").capitalize()))
    return style(fig, height)


def linked_fig(panels: list[dict]) -> go.Figure:
    """Stack panels in ONE figure with a shared x axis: zooming/panning one panel moves them all."""
    n = len(panels)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[p["title"] for p in panels])
    for i, p in enumerate(panels, start=1):
        sub = df_fig(p["df"], p["kind"], p["x"], p["ys"], p.get("color"), p["title"])
        for tr in sub.data:
            tr.legendgroup = f"p{i}"
            fig.add_trace(tr, row=i, col=1)
    fig.update_layout(template="nexus", height=max(340, 290 * n), barmode="group", showlegend=True,
                      hovermode="x unified")
    return fig
