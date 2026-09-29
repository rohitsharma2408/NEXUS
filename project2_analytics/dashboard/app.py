"""Stage 5 — Dashboards. Streamlit views over the Project 2 KPI layer."""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from features import read_sql  # noqa: E402

st.set_page_config(page_title="NEXUS — Industry Analytics", layout="wide")
st.title("NEXUS — Industry Analytics Dashboard")
st.caption("Project 2 foundation: KPIs computed directly from the PostgreSQL warehouse.")

tab_sales, tab_customers, tab_products, tab_ops = st.tabs(
    ["Sales", "Customers", "Products & Returns", "Operations"]
)

with tab_sales:
    st.subheader("Revenue & AOV by month")
    rev = read_sql("SELECT * FROM kpi_revenue_by_month")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(rev, x="month", y="total_revenue", title="Total revenue"), use_container_width=True)
    c2.plotly_chart(px.line(rev, x="month", y="aov", title="Average order value"), use_container_width=True)

    st.subheader("Revenue by country over time")
    by_country = read_sql("SELECT * FROM kpi_revenue_by_country_month")
    st.plotly_chart(
        px.line(by_country, x="month", y="revenue", color="country", title="Revenue by country"),
        use_container_width=True,
    )

with tab_customers:
    st.subheader("Customer cohorts")
    cohort = read_sql("SELECT * FROM kpi_customer_cohort")
    st.plotly_chart(
        px.bar(cohort, x="cohort_month", y="revenue", color="is_premium", title="Revenue by registration cohort"),
        use_container_width=True,
    )
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

st.divider()
st.caption(
    "Ask the AI Business Analyst instead: POST /investigate on the Project 1 API "
    f"({os.environ.get('NEXUS_API_URL', 'http://api:8000')}/docs)."
)
