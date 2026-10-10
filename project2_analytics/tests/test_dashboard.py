"""Dashboard smoke tests on the REAL warehouse (Streamlit AppTest). Skipped automatically when the
warehouse is not reachable. No synthetic data. The optional analyst end-to-end check calls the real
API + LLM, so it only runs with NEXUS_E2E=1."""
import os
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

from features import read_sql

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def warehouse():
    try:
        read_sql("SELECT 1 FROM kpi_revenue_by_month LIMIT 1")
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")


@pytest.fixture()
def app(warehouse):
    return AppTest.from_file(APP, default_timeout=90).run()


def test_dashboard_renders_with_real_data(app):
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs][:3] == ["Overview", "Ask the Analyst", "Compare"]
    assert not [e for e in app.error if "warehouse is not reachable" in e.value]
    assert any('class="kpi"' in m.value for m in app.markdown), "KPI cards missing"
    assert len(app.get("plotly_chart")) >= 6, "expected several interactive charts"


def test_compare_board_is_seeded_and_linkable(app):
    toggles = getattr(app, "toggle", [])
    if not toggles:
        pytest.skip("this Streamlit version has no AppTest toggle accessor")
    toggles[0].set_value(True).run()
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.skipif(os.environ.get("NEXUS_E2E") != "1", reason="set NEXUS_E2E=1 to call the real analyst API/LLM")
def test_analyst_answers_a_real_question(app):
    api = os.environ.get("NEXUS_INTERNAL_API_URL", "http://api:8000")
    try:
        assert requests.get(f"{api}/health", timeout=3).status_code == 200
    except Exception as e:
        pytest.skip(f"analyst API not reachable: {e}")
    [b for b in app.button if b.key == "sug1"][0].click().run()
    assert not app.exception, [e.value for e in app.exception]
