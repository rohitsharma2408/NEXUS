"""NEXUS dashboard — KPI overview, an AI analyst, and a compare board with interactive charts
side by side (optionally with a linked time axis). Streamlit, dark/gold theme."""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import read_sql  # noqa: E402
import viz  # noqa: E402

# Server-side call over the Docker network (the browser never calls this directly).
API_URL = os.environ.get("NEXUS_INTERNAL_API_URL", "http://api:8000")
DEFAULT_SPAN = (dt.date(2022, 1, 1), dt.date(2024, 12, 31))
DATE_COL = re.compile(r"^(month|year_month|order_date|cohort_month|date)$", re.I)

VIEWS = {
    "kpi_revenue_by_month": "Revenue, profit, orders and AOV by month",
    "kpi_revenue_by_country_month": "Revenue, profit and orders by country and month",
    "kpi_returns_rate_by_category": "Return rate by product category",
    "kpi_customer_cohort": "Revenue by customer registration cohort",
    "kpi_marketing_roi": "Marketing spend, revenue, ROAS and CAC by channel and month",
    "kpi_inventory_turnover": "Stock, reorder point and months of stock by product",
    "kpi_supplier_risk": "Supplier reliability, lead time and risk level",
}
SUGGESTIONS = [
    "Why did returns increase in December 2024?",
    "How did revenue in Q4 2024 compare with Q3 2024?",
    "Which product categories earned the most profit in 2024?",
    "Which suppliers are at high risk right now?",
    "Forecast demand for next month by category",
    "Were there any unusual revenue days recently?",
    "Pie chart of revenue by product category in 2024",
    "Compare monthly marketing spend with revenue by channel in 2024",
    "Which countries bring in the most profit?",
]

st.set_page_config(page_title="NEXUS — Industry Analytics", page_icon="◆", layout="wide",
                   initial_sidebar_state="expanded")
viz.register_template()

st.markdown("""
<style>
.stDeployButton, [data-testid="stToolbarActions"] {display:none;}
.block-container {padding-top: 2rem; max-width: 1500px;}
.nx-title {font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px; margin: 0;
  background: linear-gradient(90deg, #F3E5AB 0%, #D4AF37 55%, #B8860B 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.nx-sub {color: #9AA3B8; margin: 2px 0 14px 0;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
  border: 1px solid rgba(255,255,255,.15); margin-right: 6px; color:#C9D1E3;}
.pill.ok {border-color:#4FD1A5; color:#4FD1A5;} .pill.bad {border-color:#FC8181; color:#FC8181;}
.kpi {background: linear-gradient(145deg, #172039, #10162A); border: 1px solid rgba(212,175,55,.28);
  border-radius: 14px; padding: 14px 18px; height: 100%;}
.kpi .label {color:#9AA3B8; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi .value {font-size: 1.85rem; font-weight: 750; color:#F4F1E8; line-height: 1.25;}
.kpi .delta {font-size:.85rem; font-weight:600;} .kpi .delta em {color:#7C869C; font-weight:400; font-style:normal;}
.kpi .good {color:#4FD1A5;} .kpi .bad {color:#FC8181;}
.badge {display:inline-block; padding: 2px 12px; border-radius: 8px; font-weight:700; font-size:.82rem;}
.badge.high {background:rgba(79,209,165,.15); color:#4FD1A5;}
.badge.medium {background:rgba(246,173,85,.15); color:#F6AD55;}
.badge.low {background:rgba(252,129,129,.15); color:#FC8181;}
.muted {color:#7C869C; font-size:.85rem;}
div[data-baseweb="tab-list"] {gap: 6px;}
div[data-baseweb="tab-list"] button[aria-selected="true"] {color:#D4AF37;}
</style>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("board", [])
ss.setdefault("pending_q", None)
ss.setdefault("seeded", False)
ss.setdefault("next_id", 1)


# ------------------------------------------------------------------------------ data access
@st.cache_data(ttl=30, show_spinner=False)
def warehouse_status() -> tuple[bool, str]:
    try:
        read_sql("SELECT 1 AS ok")
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0][:200]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Postgres numeric -> float, timestamptz -> naive, so arithmetic and date filters just work."""
    df = df.copy()
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            df[c] = s.dt.tz_localize(None)
        elif s.dtype == object:
            first = s.dropna().head(1)
            if len(first) and first.iloc[0].__class__.__name__ == "Decimal":
                df[c] = pd.to_numeric(s, errors="coerce")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load(view: str) -> pd.DataFrame:
    return _normalise(read_sql(f"SELECT * FROM {view}"))  # view is whitelisted in load()


def load(view: str) -> pd.DataFrame | None:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view}")
    try:
        return _load(view)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=30, show_spinner=False)
def api_status() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def apply_filters(df: pd.DataFrame | None, f: dict) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    out = df
    date_col = next((c for c in out.columns if DATE_COL.match(c)), None)
    if date_col:
        d = pd.to_datetime(out[date_col], errors="coerce")
        out = out[(d >= pd.Timestamp(f["start"])) & (d <= pd.Timestamp(f["end"]))]
    if "country" in out.columns and f["countries"]:
        out = out[out["country"].isin(f["countries"])]
    if "category" in out.columns and f["categories"]:
        out = out[out["category"].isin(f["categories"])]
    return out


def human(v: float, money: bool = False) -> str:
    p = "$" if money else ""
    for lim, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= lim:
            return f"{p}{v / lim:,.2f}{suf}"
    return f"{p}{v:,.0f}" if abs(v) >= 100 else f"{p}{v:,.2f}"


def kpi_card(label: str, value: str, delta: float | None = None, good_up: bool = True, unit: str = "%") -> str:
    d = ""
    if delta is not None and pd.notna(delta):
        up = delta >= 0
        cls = "good" if up == good_up else "bad"
        d = f'<div class="delta {cls}">{"▲" if up else "▼"} {abs(delta):.1f}{unit} <em>vs prior month</em></div>'
    return f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>'


def show(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config=viz.PLOT_CONFIG, key=key)


def table(df: pd.DataFrame, name: str, key: str) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=key)


def empty(msg: str = "No data for the current filters.") -> None:
    st.info(msg)


# ------------------------------------------------------------------------------ header + sidebar
wh_ok, wh_err = warehouse_status()
api_ok = api_status()

st.markdown('<p class="nx-title">NEXUS — Industry Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="nx-sub">Live KPIs from the PostgreSQL warehouse, an AI analyst you can question, '
            'and a board for comparing interactive charts side by side.</p>', unsafe_allow_html=True)
st.markdown(
    f'<span class="pill {"ok" if wh_ok else "bad"}">Warehouse {"connected" if wh_ok else "offline"}</span>'
    f'<span class="pill {"ok" if api_ok else "bad"}">Analyst API {"online" if api_ok else "offline"}</span>',
    unsafe_allow_html=True)
if not wh_ok:
    st.error(f"The warehouse is not reachable, so charts cannot load: {wh_err}")

rev_all = load("kpi_revenue_by_month") if wh_ok else None
by_country_all = load("kpi_revenue_by_country_month") if wh_ok else None
returns_all = load("kpi_returns_rate_by_category") if wh_ok else None

span = DEFAULT_SPAN
if rev_all is not None and not rev_all.empty:
    m = pd.to_datetime(rev_all["month"])
    span = (m.min().date(), m.max().date())

with st.sidebar:
    st.markdown("### ◆ Filters")
    picked = st.date_input("Period", value=span, min_value=span[0], max_value=span[1])
    start, end = picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else span
    countries = st.multiselect(
        "Countries", sorted(by_country_all["country"].dropna().unique()) if by_country_all is not None else [],
        placeholder="All countries")
    categories = st.multiselect(
        "Categories", sorted(returns_all["category"].dropna().unique()) if returns_all is not None else [],
        placeholder="All categories")
    st.caption("Filters apply to every tab and to view-based charts on the Compare board.")
    st.divider()
    st.markdown(f"<span class='muted'>Data covers {span[0]:%b %Y} – {span[1]:%b %Y}.</span>",
                unsafe_allow_html=True)

FILTERS = {"start": start, "end": end, "countries": countries, "categories": categories}

tab_over, tab_ask, tab_cmp, tab_sales, tab_cust, tab_prod, tab_ops = st.tabs(
    ["Overview", "Ask the Analyst", "Compare", "Sales", "Customers", "Products & Returns", "Operations"])


# ------------------------------------------------------------------------------ Overview
with tab_over:
    rev = apply_filters(rev_all, FILTERS)
    if rev is None or rev.empty:
        empty("Revenue data is unavailable for the current filters.")
    else:
        rev = rev.assign(month=pd.to_datetime(rev["month"])).sort_values("month")
        last, prev = rev.iloc[-1], (rev.iloc[-2] if len(rev) > 1 else None)

        def chg(col: str):
            return None if prev is None or not prev[col] else (last[col] / prev[col] - 1) * 100

        margin = last["total_profit"] / last["total_revenue"] * 100 if last["total_revenue"] else float("nan")
        margin_prev = (prev["total_profit"] / prev["total_revenue"] * 100
                       if prev is not None and prev["total_revenue"] else None)
        st.caption(f"Latest month in view: **{last['month']:%B %Y}**")
        c = st.columns(5)
        c[0].markdown(kpi_card("Revenue", human(last["total_revenue"], True), chg("total_revenue")), unsafe_allow_html=True)
        c[1].markdown(kpi_card("Profit", human(last["total_profit"], True), chg("total_profit")), unsafe_allow_html=True)
        c[2].markdown(kpi_card("Orders", human(last["order_count"]), chg("order_count")), unsafe_allow_html=True)
        c[3].markdown(kpi_card("Avg order value", human(float(last["aov"]), True), chg("aov")), unsafe_allow_html=True)
        c[4].markdown(kpi_card("Profit margin", f"{margin:.1f}%",
                               None if margin_prev is None else margin - margin_prev, unit=" pts"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(rev, "line", "month", ["total_revenue", "total_profit"], title="Revenue and profit"), "ov_rp")
        with b:
            show(viz.df_fig(rev, "bar", "month", ["order_count"], title="Orders per month"), "ov_orders")
        a, b = st.columns(2)
        bc = apply_filters(by_country_all, FILTERS)
        with a:
            if bc is not None and not bc.empty:
                top = bc.groupby("country")["revenue"].sum().nlargest(6).index
                show(viz.df_fig(bc[bc["country"].isin(top)], "line", "month", ["revenue"], "country",
                                "Revenue by country (top 6)"), "ov_country")
            else:
                empty()
        with b:
            rr = apply_filters(returns_all, FILTERS)
            if rr is not None and not rr.empty:
                show(viz.df_fig(rr.sort_values("return_rate_pct"), "bar", "category", ["return_rate_pct"],
                                title="Return rate by category (%)"), "ov_returns")
            else:
                empty()


# ------------------------------------------------------------------------------ Ask the Analyst
def run_question(q: str) -> dict:
    t0 = time.time()
    try:
        resp = requests.post(f"{API_URL}/investigate", json={"question": q}, timeout=180)
    except requests.RequestException as e:
        return {"question": q, "error": f"Could not reach the analyst API: {e}"}
    if resp.status_code != 200:
        return {"question": q, "error": f"The analyst API returned {resp.status_code}: {resp.text[:600]}"}
    data = resp.json()
    data["question"], data["elapsed"] = q, round(time.time() - t0, 1)
    return data


def md_safe(text: str) -> str:
    """Streamlit renders $...$ as LaTeX, which garbles money amounts. Escape every dollar sign."""
    return (text or "").replace("\\$", "$").replace("$", "\\$")


def pin(panel: dict, title: str) -> None:
    panel["id"], panel["title"] = ss.next_id, title
    ss.next_id += 1
    ss.board.append(panel)
    st.toast(f"Pinned “{title}” to the Compare board")


def render_answer(item: dict, idx: int) -> None:
    with st.container(border=True):
        st.markdown(f"**{md_safe(item['question'])}**")
        if item.get("error"):
            st.error(item["error"])
            st.caption("If this mentions the LLM, check LLM_PROVIDER, LLM_MODEL and the matching API key in .env.")
            return
        conf = str(item.get("confidence", "")).lower()
        ver = item.get("verification") or {}
        bits = [f'<span class="badge {conf}">{conf.capitalize() or "n/a"} confidence</span>']
        if "checked" in ver:
            bits.append(f'<span class="muted">{ver.get("checked", 0)} figures checked against the data, '
                        f'{len(ver.get("unsupported", []))} unsupported</span>')
        if item.get("elapsed"):
            bits.append(f'<span class="muted">{item["elapsed"]}s</span>')
        st.markdown(" &nbsp; ".join(bits), unsafe_allow_html=True)
        st.markdown(md_safe(item.get("answer", "")))
        for cav in item.get("caveats", []):
            st.warning(md_safe(cav))

        specs = item.get("charts") or []
        if specs:
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                with cols[i % 2]:
                    show(viz.spec_to_fig(spec, 340), f"ans_{idx}_{i}")
                    if st.button("📌 Pin to Compare", key=f"pin_{idx}_{i}"):
                        pin({"spec": spec}, spec["title"])

        ev = item.get("evidence") or {}
        sql_ev = ev.get("sql") or {}
        rows = sql_ev.get("rows") or []
        dd = ev.get("drilldown") or {}
        chunks = (ev.get("rag") or {}).get("chunks", [])
        t1, t2, t3, t4 = st.tabs(["Drill-down", "Data & SQL", "Documents", "How it was answered"])
        with t1:
            if dd:
                st.write(f"**{dd['metric']}**: {dd['baseline_value']:,.0f} → {dd['target_value']:,.0f} "
                         f"({dd['total_change_pct']}%)")
                dims = list((dd.get("by_dimension") or {}).items())
                for chunk in range(0, len(dims), 3):
                    cols = st.columns(min(3, len(dims) - chunk))
                    for col, (dim, items) in zip(cols, dims[chunk:chunk + 3]):
                        col.markdown(f"**By {dim}**")
                        col.dataframe(items, use_container_width=True, hide_index=True)
            else:
                st.caption("No period-over-period drill-down for this question.")
        with t2:
            if rows:
                df = pd.DataFrame(rows)
                table(df, "analyst_rows", f"dl_{idx}")
                num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if len(df) >= 2 and num:
                    st.markdown("**Chart these rows yourself**")
                    c1, c2, c3 = st.columns(3)
                    x = c1.selectbox("X axis", list(df.columns), key=f"x_{idx}")
                    ys = c2.multiselect("Values", num, default=num[:1], key=f"y_{idx}")
                    kind = c3.selectbox("Type", viz.KINDS, key=f"k_{idx}")
                    if ys:
                        title = f"{', '.join(ys)} by {x}"
                        show(viz.df_fig(df, kind, x, ys, title=title), f"own_{idx}")
                        if st.button("📌 Pin to Compare", key=f"ownpin_{idx}"):
                            pin({"df": df, "kind": kind, "x": x, "ys": ys, "color": None}, title)
            else:
                st.caption("The query returned no rows.")
            if sql_ev.get("sql"):
                st.code(sql_ev["sql"], language="sql")
        with t3:
            if chunks:
                for ch in chunks:
                    st.markdown(f"**{ch['document']}** (similarity {ch['score']:.2f})")
                    st.text(ch["content"])
            else:
                st.caption("No supporting documents were used.")
        with t4:
            st.json({"routing": item.get("routing"), "verification": ver,
                     "sql_rejected": sql_ev.get("rejected_reason")})


with tab_ask:
    st.markdown("### Ask the AI Business Analyst")
    st.caption(f"Data covers {span[0]:%b %Y} – {span[1]:%b %Y}. Name the period you mean "
               "(e.g. “December 2024”) for the sharpest answers.")
    with st.form("ask_form"):
        c1, c2 = st.columns([6, 1])
        q_in = c1.text_input("Your question", placeholder="Why did returns increase in December 2024?",
                             label_visibility="collapsed")
        submitted = c2.form_submit_button("Investigate", type="primary", use_container_width=True)
    if submitted and q_in.strip():
        ss.pending_q = q_in.strip()
    st.markdown('<span class="muted">Or try one:</span>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 3].button(s, key=f"sug{i}", use_container_width=True):
            ss.pending_q = s
    if ss.pending_q:
        question, ss.pending_q = ss.pending_q, None
        with st.spinner("Investigating — querying the warehouse, checking the numbers (10–40 s)…"):
            ss.history.insert(0, run_question(question))
    for idx, item in enumerate(ss.history):
        render_answer(item, idx)
    if ss.history and st.button("Clear conversation"):
        ss.history = []
        st.rerun()


# ------------------------------------------------------------------------------ Compare board
def resolve(panel: dict):
    """-> (dataframe or None, figure) for a board panel. View panels follow the sidebar filters."""
    if "spec" in panel:
        return None, viz.spec_to_fig(panel["spec"])
    df = apply_filters(load(panel["view"]), FILTERS) if "view" in panel else panel["df"]
    if df is None or df.empty:
        return df, None
    return df, viz.df_fig(df, panel["kind"], panel["x"], panel["ys"], panel.get("color"), panel["title"])


def add_view_panel(view, kind, x, ys, color, title) -> None:
    ss.board.append({"id": ss.next_id, "title": title, "view": view, "kind": kind,
                     "x": x, "ys": ys, "color": color})
    ss.next_id += 1


with tab_cmp:
    st.markdown("### Compare board")
    st.caption("Build or pin charts, put them side by side, and link the time axis so one zoom moves them all.")
    if wh_ok and not ss.seeded and rev_all is not None and not ss.board:
        add_view_panel("kpi_revenue_by_month", "line", "month", ["total_revenue", "total_profit"], None,
                       "Revenue and profit by month")
        if by_country_all is not None:
            add_view_panel("kpi_revenue_by_country_month", "line", "month", ["revenue"], "country",
                           "Revenue by country")
        ss.seeded = True

    c1, c2, c3 = st.columns([2, 2, 1])
    per_row = c1.radio("Charts per row", [1, 2, 3], index=1, horizontal=True)
    linked = c2.toggle("Link time axes (zoom one, zoom all)", value=False)
    if c3.button("Clear board", use_container_width=True):
        ss.board = []
        ss.seeded = True
        st.rerun()

    with st.expander("➕ Add a chart from the warehouse", expanded=not ss.board):
        if not wh_ok:
            empty("The warehouse is offline.")
        else:
            view = st.selectbox("Data", list(VIEWS), format_func=lambda v: f"{v} — {VIEWS[v]}", key="b_view")
            src = load(view)
            if src is None or src.empty:
                empty("That view has no rows.")
            else:
                cols_all = list(src.columns)
                num = [c for c in cols_all if pd.api.types.is_numeric_dtype(src[c])]
                cat = [c for c in cols_all if c not in num]
                x_default = next((c for c in cols_all if DATE_COL.match(c)), cat[0] if cat else cols_all[0])
                b1, b2, b3, b4 = st.columns(4)
                x = b1.selectbox("X axis", cols_all, index=cols_all.index(x_default), key="b_x")
                ys = b2.multiselect("Values", num, default=num[:1], key="b_y")
                color = b3.selectbox("Split by", ["(none)"] + [c for c in cat if c != x], key="b_c")
                kind = b4.selectbox("Type", viz.KINDS, key="b_k")
                title = st.text_input("Title", value=(f"{', '.join(ys)} by {x}" if ys else ""), key="b_t")
                if st.button("Add to board", type="primary", disabled=not ys):
                    add_view_panel(view, kind, x, ys, None if color == "(none)" else color, title or view)
                    st.rerun()

    if not ss.board:
        empty("The board is empty. Add a chart above, or pin one from an analyst answer.")
    else:
        resolved = [(p, *resolve(p)) for p in ss.board]
        to_link = []
        if linked:
            to_link = [(p, df) for p, df, fig in resolved
                       if "spec" not in p and df is not None and not df.empty and viz.is_temporal(df, p["x"])]
            if len(to_link) >= 2:
                lp = [{"df": df, "kind": p["kind"], "x": p["x"], "ys": p["ys"], "color": p.get("color"),
                       "title": p["title"]} for p, df in to_link]
                show(viz.linked_fig(lp), "linked_fig")
                st.caption("Linked view: panels share one time axis. Use the modebar or scroll to zoom; "
                           "double-click to reset.")
            else:
                st.info("Linking needs at least two panels with a date on the X axis; showing them separately.")
                to_link = []
        linked_ids = {p["id"] for p, _ in to_link}
        rest = [(p, df, fig) for p, df, fig in resolved if p["id"] not in linked_ids]
        for r in range(0, len(rest), per_row):
            cols = st.columns(per_row)
            for col, (p, df, fig) in zip(cols, rest[r:r + per_row]):
                with col, st.container(border=True):
                    if fig is None:
                        empty()
                    else:
                        show(fig, f"board_{p['id']}")
                    b1, b2 = st.columns(2)
                    if b1.button("Remove", key=f"rm_{p['id']}", use_container_width=True):
                        ss.board = [x for x in ss.board if x["id"] != p["id"]]
                        st.rerun()
                    if df is not None and not df.empty:
                        b2.download_button("CSV", df.to_csv(index=False).encode(), f"{p['id']}.csv",
                                           "text/csv", key=f"csv_{p['id']}", use_container_width=True)
        if linked and to_link:
            if st.button("Remove linked panels"):
                ss.board = [x for x in ss.board if x["id"] not in linked_ids]
                st.rerun()


# ------------------------------------------------------------------------------ Sales
with tab_sales:
    rev = apply_filters(rev_all, FILTERS)
    bc = apply_filters(by_country_all, FILTERS)
    if rev is None or rev.empty:
        empty()
    else:
        rev = rev.assign(month=pd.to_datetime(rev["month"]), margin_pct=lambda d: d["total_profit"] / d["total_revenue"] * 100)
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(rev, "area", "month", ["total_revenue"], title="Total revenue"), "s_rev")
        with b:
            show(viz.df_fig(rev, "line", "month", ["margin_pct"], title="Profit margin (%)"), "s_margin")
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(rev, "line", "month", ["aov"], title="Average order value"), "s_aov")
        with b:
            if bc is not None and not bc.empty:
                tot = bc.groupby("country", as_index=False)["revenue"].sum().sort_values("revenue")
                show(viz.df_fig(tot, "bar", "country", ["revenue"], title="Revenue by country (selected period)"), "s_ctry_bar")
        if bc is not None and not bc.empty:
            show(viz.df_fig(bc, "area", "month", ["revenue"], "country", "Revenue by country over time", 420), "s_ctry_area")
        with st.expander("Data table"):
            table(rev, "revenue_by_month", "dl_sales")


# ------------------------------------------------------------------------------ Customers
with tab_cust:
    cohort = apply_filters(load("kpi_customer_cohort") if wh_ok else None, FILTERS)
    if cohort is None or cohort.empty:
        empty()
    else:
        cohort = cohort.assign(cohort_month=pd.to_datetime(cohort["cohort_month"]),
                               is_premium=cohort["is_premium"].astype(str))
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(cohort, "bar", "cohort_month", ["revenue"], "is_premium",
                            "Revenue by registration cohort (premium vs not)"), "c_rev")
        with b:
            show(viz.df_fig(cohort, "line", "cohort_month", ["customers"], "is_premium",
                            "New customers per cohort"), "c_cust")
        with st.expander("Data table"):
            table(cohort, "customer_cohort", "dl_cust")


# ------------------------------------------------------------------------------ Products & returns
with tab_prod:
    rr = apply_filters(returns_all, FILTERS)
    if rr is None or rr.empty:
        empty()
    else:
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(rr.sort_values("return_rate_pct"), "bar", "category", ["return_rate_pct"],
                            title="Return rate by category (%)"), "p_rate")
        with b:
            sc = px.scatter(rr, x="total_sales", y="return_rate_pct", size="return_count", color="category",
                            hover_name="category", title="Sales volume vs return rate", template="nexus")
            show(viz.style(sc), "p_scatter")
        with st.expander("Data table"):
            table(rr, "returns_by_category", "dl_prod")


# ------------------------------------------------------------------------------ Operations
with tab_ops:
    inv = apply_filters(load("kpi_inventory_turnover") if wh_ok else None, FILTERS)
    sup = load("kpi_supplier_risk") if wh_ok else None
    mkt = apply_filters(load("kpi_marketing_roi") if wh_ok else None, FILTERS)
    if inv is not None and not inv.empty:
        inv = inv.assign(at_reorder_risk=inv["at_reorder_risk"].fillna(False).astype(bool))
    st.markdown("#### Inventory")
    if inv is None or inv.empty:
        empty()
    else:
        a, b = st.columns(2)
        with a:
            show(viz.style(px.histogram(inv, x="months_of_stock", nbins=30, title="Months of stock on hand",
                                        template="nexus")), "o_hist")
        with b:
            risk = inv[inv["at_reorder_risk"]].groupby("category", as_index=False).size().rename(columns={"size": "products"})
            if risk.empty:
                empty("No products are at reorder risk.")
            else:
                show(viz.df_fig(risk.sort_values("products"), "bar", "category", ["products"],
                                title="Products at reorder risk, by category"), "o_risk")
        with st.expander("Products at reorder risk"):
            table(inv[inv["at_reorder_risk"]], "reorder_risk", "dl_inv")
    st.markdown("#### Supplier risk")
    if sup is None or sup.empty:
        empty()
    else:
        a, b = st.columns(2)
        with a:
            lv = sup["supplier_risk_level"].value_counts().rename_axis("level").reset_index(name="suppliers")
            pie = px.pie(lv, names="level", values="suppliers", hole=0.55, title="Suppliers by risk level",
                         color="level", color_discrete_map=viz.LEVEL_COLORS, template="nexus")
            show(viz.style(pie), "o_pie")
        with b:
            sc = px.scatter(sup, x="lead_time_days", y="reliability_score", color="supplier_risk_level",
                            color_discrete_map=viz.LEVEL_COLORS, hover_name="supplier_name",
                            title="Reliability vs lead time", template="nexus")
            show(viz.style(sc), "o_sup")
        with st.expander("Supplier table"):
            order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            table(sup.sort_values("supplier_risk_level", key=lambda s: s.map(order)), "supplier_risk", "dl_sup")
    st.markdown("#### Marketing")
    if mkt is None or mkt.empty:
        empty()
    else:
        a, b = st.columns(2)
        with a:
            show(viz.df_fig(mkt, "line", "year_month", ["roas"], "channel", "Return on ad spend (ROAS)"), "o_roas")
        with b:
            show(viz.df_fig(mkt, "line", "year_month", ["cac_usd"], "channel", "Customer acquisition cost ($)"), "o_cac")
