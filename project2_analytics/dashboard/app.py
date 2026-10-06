"""Stage 5 — Dashboards. Streamlit KPI views + an Ask tab wired to the Project 1 API."""
import os
import sys
from pathlib import Path

import plotly.express as px
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import read_sql  # noqa: E402

# Server-side call over the Docker network (the browser never calls this directly).
API_URL = os.environ.get("NEXUS_INTERNAL_API_URL", "http://api:8000")

st.set_page_config(page_title="NEXUS — Industry Analytics", layout="wide")
st.title("NEXUS — Industry Analytics Dashboard")
st.caption("KPIs computed directly from the PostgreSQL warehouse, plus an AI analyst you can question.")

tab_ask, tab_sales, tab_customers, tab_products, tab_ops = st.tabs(
    ["Ask the Analyst", "Sales", "Customers", "Products & Returns", "Operations"]
)

with tab_ask:
    st.subheader("Ask the AI Business Analyst")
    st.caption("The data covers Jan 2022 – Dec 2024, so ask about specific periods (e.g. 'December 2024').")
    question = st.text_input("Your question", placeholder="Why did returns increase in December?")
    if st.button("Investigate") and question.strip():
        resp = None
        with st.spinner("Investigating (this can take 10-30 seconds)..."):
            try:
                resp = requests.post(f"{API_URL}/investigate", json={"question": question}, timeout=180)
            except requests.RequestException as e:
                st.error(f"Could not reach the API: {e}")
        if resp is not None:
            if resp.status_code != 200:
                st.error(f"API returned {resp.status_code}: {resp.text[:600]}")
            else:
                data = resp.json()
                st.markdown(data["answer"])
                st.metric("Confidence", str(data["confidence"]).capitalize())
                for caveat in data.get("caveats", []):
                    st.warning(caveat)
                for spec in data.get("charts", []):
                    fig = px.line if spec["type"] == "line" else px.bar
                    long = [{"x": x, "series": ser["name"], "y": y}
                            for ser in spec["series"] for x, y in zip(spec["x"], ser["y"])]
                    st.plotly_chart(
                        fig(long, x="x", y="y", color="series", title=spec["title"],
                            labels={"x": spec["x_label"], "y": spec["y_label"]}, **(
                                {"markers": True} if spec["type"] == "line" else {"barmode": "group"})),
                        use_container_width=True)
                dd = data["evidence"].get("drilldown") or {}
                if dd:
                    with st.expander("Drill-down: what drove the change"):
                        st.write(f"{dd['metric']}: {dd['baseline_value']:,.0f} to {dd['target_value']:,.0f} "
                                 f"({dd['total_change_pct']}%)")
                        for dim, items in dd["by_dimension"].items():
                            st.markdown(f"**By {dim}**")
                            st.dataframe(items)
                sql_ev = data["evidence"]["sql"]
                with st.expander("SQL used"):
                    st.code(sql_ev.get("sql", ""), language="sql")
                with st.expander("Rows returned"):
                    st.dataframe(sql_ev.get("rows", []))
                chunks = data["evidence"]["rag"].get("chunks", [])
                with st.expander(f"Document evidence ({len(chunks)} chunks)"):
                    for c in chunks:
                        st.markdown(f"**{c['document']}** (similarity {c['score']:.2f})")
                        st.text(c["content"])

with tab_sales:
    st.subheader("Revenue & AOV by month")
    rev = read_sql("SELECT * FROM kpi_revenue_by_month")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(rev, x="month", y="total_revenue", title="Total revenue"), use_container_width=True)
    c2.plotly_chart(px.line(rev, x="month", y="aov", title="Average order value"), use_container_width=True)
    st.subheader("Revenue by country over time")
    by_country = read_sql("SELECT * FROM kpi_revenue_by_country_month")
    st.plotly_chart(px.line(by_country, x="month", y="revenue", color="country", title="Revenue by country"),
                    use_container_width=True)

with tab_customers:
    st.subheader("Customer cohorts")
    cohort = read_sql("SELECT * FROM kpi_customer_cohort")
    st.plotly_chart(px.bar(cohort, x="cohort_month", y="revenue", color="is_premium",
                           title="Revenue by registration cohort"), use_container_width=True)
    st.dataframe(cohort)

with tab_products:
    st.subheader("Returns rate by category")
    returns = read_sql("SELECT * FROM kpi_returns_rate_by_category")
    st.plotly_chart(px.bar(returns, x="category", y="return_rate_pct"), use_container_width=True)
    st.dataframe(returns)

with tab_ops:
    st.subheader("Inventory turnover / reorder risk")
    inv = read_sql("SELECT * FROM kpi_inventory_turnover")
    st.dataframe(inv[inv["at_reorder_risk"]])
    st.plotly_chart(px.histogram(inv, x="months_of_stock", nbins=30), use_container_width=True)
    st.subheader("Supplier risk")
    sup = read_sql("SELECT * FROM kpi_supplier_risk")
    st.dataframe(sup.sort_values("supplier_risk_level"))
    st.subheader("Marketing ROI")
    mkt = read_sql("SELECT * FROM kpi_marketing_roi")
    st.plotly_chart(px.line(mkt, x="year_month", y="roas", color="channel"), use_container_width=True)
