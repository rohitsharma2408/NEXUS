"""Chart generation tests on REAL rows pulled from the warehouse (KPI views / fact tables)."""
import json
import os
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

import charts

RO = os.environ["READONLY_DATABASE_URL"]


def q(sql):
    try:
        with create_engine(RO).connect() as c:
            res = c.execute(text(sql))
            return [dict(zip(res.keys(), r)) for r in res.fetchall()]
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")


def inv(rows=None, ml=None, drilldown=None):
    return SimpleNamespace(sql_findings={"rows": rows or []}, ml_findings=ml or {}, drilldown=drilldown or {})


@pytest.fixture(scope="module")
def monthly():
    return q("SELECT month, total_revenue, order_count FROM kpi_revenue_by_month ORDER BY month DESC LIMIT 36")  # newest first on purpose


@pytest.fixture(scope="module")
def by_category():
    return q("SELECT category, return_rate_pct FROM kpi_returns_rate_by_category")


def test_real_monthly_kpi_becomes_a_sorted_line_chart(monthly):
    c = charts.chart_from_rows(monthly)
    assert c["type"] == "line" and len(c["x"]) == 36 and c["x"] == sorted(c["x"])
    assert c["x"][0] == "2022-01-01" and c["x"][-1] == "2024-12-01"
    assert [s["name"] for s in c["series"]] == ["Total revenue", "Order count"]
    # values line up with the rows they came from
    by_month = {r["month"].strftime("%Y-%m-%d"): float(r["total_revenue"]) for r in monthly}
    assert c["series"][0]["y"] == [by_month[x] for x in c["x"]]


def test_real_category_rates_become_a_bar_chart(by_category):
    c = charts.chart_from_rows(by_category)
    assert c["type"] == "bar" and set(c["x"]) == {r["category"] for r in by_category} and len(c["x"]) == 7


def test_not_chartable_cases(monthly):
    assert charts.chart_from_rows([]) is None
    assert charts.chart_from_rows(monthly[:1]) is None                                   # one row
    ids = q("SELECT product_id, supplier_name FROM fact_supplier_costs LIMIT 5")
    assert charts.chart_from_rows(ids) is None                                          # nothing numeric
    many = q("SELECT product_id, SUM(quantity) AS units FROM fact_sales GROUP BY product_id LIMIT 60")
    assert charts.chart_from_rows(many) is None                                         # too many bars


def test_id_can_be_a_label_when_values_are_numeric():
    top = q("SELECT product_id, SUM(quantity) AS units FROM fact_sales GROUP BY product_id ORDER BY 2 DESC LIMIT 5")
    c = charts.chart_from_rows(top)
    assert c["type"] == "bar" and c["x"] == [r["product_id"] for r in top]


def test_spec_is_json_serialisable_and_converts_to_plotly(monthly):
    c = charts.chart_from_rows(monthly)
    json.dumps(c)
    fig = charts.spec_to_plotly_dict(c)
    assert fig["data"][0]["type"] == "scatter" and fig["layout"]["title"]["text"] == c["title"]


def test_drilldown_and_forecast_charts_from_real_playbook_output(monthly):
    import investigation_playbook as pb
    dd = pb.investigate_change("Why did revenue fall in February 2023?", create_engine(RO))
    ml = {"demand_forecast": {"forecast_month": "2025-01",
                              "predicted_units_by_category": {"Clothing": 1245.0, "Books": 465.0}}}
    out = charts.build_charts(inv(rows=monthly, ml=ml, drilldown=dd))
    assert [c["type"] for c in out] == ["bar", "bar", "line"] and len(out) <= charts.MAX_CHARTS
    assert out[0]["series"][0]["y"] == [i["change"] for i in dd["by_dimension"]["country"]]


def test_only_values_from_the_evidence_are_plotted(monthly):
    out = charts.build_charts(inv(rows=monthly))
    allowed = {float(r[k]) for r in monthly for k in ("total_revenue", "order_count")}
    assert {y for s in out[0]["series"] for y in s["y"]} <= allowed


# ---- multi-row-per-month data, and pie requests (real warehouse rows) ------------------------
@pytest.fixture(scope="module")
def marketing():
    return q("SELECT year_month, channel, spend_usd FROM kpi_marketing_roi ORDER BY year_month, channel")


def test_real_channel_per_month_rows_become_one_line_per_channel(marketing):
    c = charts.chart_from_rows(marketing)
    assert c["type"] == "line"
    assert {s["name"] for s in c["series"]} == {r["channel"] for r in marketing}
    assert len(set(c["x"])) == len(c["x"]) and c["x"] == sorted(c["x"])      # one point per month: no zigzag
    assert all(len(s["y"]) == len(c["x"]) for s in c["series"])
    plotted = sum(v for s in c["series"] for v in s["y"] if v is not None)
    assert abs(plotted - sum(float(r["spend_usd"]) for r in marketing)) < 0.01   # nothing lost or invented


def test_real_repeated_months_without_a_category_are_summed():
    rows = q("SELECT year_month, spend_usd, actual_revenue_usd FROM kpi_marketing_roi ORDER BY year_month")
    c = charts.chart_from_rows(rows)
    assert len(set(c["x"])) == len(c["x"])
    assert abs(sum(c["series"][0]["y"]) - sum(float(r["spend_usd"]) for r in rows)) < 0.01


def test_pie_request_on_real_monthly_revenue_keeps_the_total(monthly):
    c = charts.chart_from_rows(monthly, prefer_pie=True)
    assert c["type"] == "pie" and len(c["x"]) <= charts.MAX_PIE_SLICES
    assert c["x"][-1].startswith("Other")                                   # 36 months folded into 12 slices
    assert abs(sum(c["series"][0]["y"]) - sum(float(r["total_revenue"]) for r in monthly)) < 0.01


def test_build_charts_puts_the_requested_pie_first(monthly):
    investigation = inv(rows=monthly)
    investigation.question = "give me the pie chart of monthly sales"
    out = charts.build_charts(investigation)
    assert out[0]["type"] == "pie" and any(ch["type"] == "line" for ch in out[1:])
    assert charts.spec_to_plotly_dict(out[0])["data"][0]["type"] == "pie"


def test_pie_of_real_categories(by_category):
    c = charts.chart_from_rows(by_category, prefer_pie=True)
    assert c["type"] == "pie" and set(c["x"]) == {r["category"] for r in by_category}
