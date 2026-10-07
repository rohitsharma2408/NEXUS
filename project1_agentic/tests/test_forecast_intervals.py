"""The forecast agent's 80% ranges must bracket the point forecast. Skipped if the calibration file
(evaluation/forecast_interval_calibration.json) has not been generated."""
import pytest

import ml_agent


def test_ranges_bracket_the_point_forecast():
    s = ml_agent.forecast_next_month().summary
    if "category_ranges_80" not in s:
        pytest.skip("interval calibration file not present")
    for r in s["predictions"]:
        assert 0 <= r["low_80"] <= r["predicted_next_month_units"] <= r["high_80"]
    for cat, (lo, hi) in s["category_ranges_80"].items():
        assert lo <= s["predicted_units_by_category"][cat] <= hi
