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
