"""Tests for the forecasting and anomaly cores, run on the REAL warehouse data (fact_price_history,
fact_sales, fact_returns). Skipped automatically when the warehouse is not reachable.

They guard the two things that made the old models untrustworthy: information leakage and
missing seasonality."""
import numpy as np
import pandas as pd
import pytest

import anomaly_core as ac
import forecast_core as fc
from features import read_sql


@pytest.fixture(scope="module")
def panel():
    try:
        raw = read_sql("""SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
                                 competitor_price_usd, price_index, is_promotional, units_sold
                          FROM fact_price_history ORDER BY product_id, year_month""")
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")
    return fc.build_panel(raw)


@pytest.fixture(scope="module")
def daily():
    try:
        sales = read_sql("SELECT order_date, SUM(revenue_usd) AS revenue, COUNT(*) AS orders "
                         "FROM fact_sales GROUP BY order_date ORDER BY order_date")
        ret = read_sql("SELECT return_date AS order_date, COUNT(*) AS return_count FROM fact_returns GROUP BY return_date")
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")
    d = sales.merge(ret, on="order_date", how="left")
    d["order_date"] = pd.to_datetime(d["order_date"])
    d["return_count"] = d["return_count"].fillna(0)
    return d.set_index("order_date").sort_index()[["revenue", "orders", "return_count"]]


FIRST_2023 = int(2023 * 12)          # month index of 2023-01


# ------------------------------------------------------------------ forecasting
def test_month_index_roundtrip():
    assert fc.idx_to_ym(int(fc.ym_to_idx(pd.Series(["2024-03"]))[0])) == "2024-03"
    assert fc.idx_to_ym(2022 * 12 + 11) == "2022-12"


def test_real_panel_shape(panel):
    assert panel["product_id"].nunique() == 500 and panel["year_month"].min() == "2022-01" \
        and panel["year_month"].max() == "2024-12"


def test_features_use_only_information_up_to_origin_month(panel):
    """Corrupt every month after the cut-off; features for origins up to the cut-off must not change."""
    cut = FIRST_2023 + 8
    season = fc.SeasonalIndex.fit(panel, upto_idx=cut)
    f1 = fc.make_features(panel, season)
    p2 = panel.copy()
    p2.loc[p2["midx"] > cut, ["units_sold", "price_index"]] = [9999.0, 9.9]
    f2 = fc.make_features(p2, season)
    a = f1[f1["midx"] <= cut].drop(columns=["target"]).reset_index(drop=True)
    b = f2[f2["midx"] <= cut].drop(columns=["target"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_target_is_next_month_units(panel):
    f = fc.make_features(panel, fc.SeasonalIndex.fit(panel))
    r = f[(f["product_id"] == "PROD0055") & (f["year_month"] == "2023-05")].iloc[0]
    nxt = panel[(panel["product_id"] == "PROD0055") & (panel["year_month"] == "2023-06")]["units_sold"].iloc[0]
    assert r["target"] == nxt


def test_seasonal_index_ignores_future_and_matches_real_pattern(panel):
    cut = FIRST_2023 + 11                      # fit on 2022-01..2023-12 only
    s1 = fc.SeasonalIndex.fit(panel, upto_idx=cut)
    p2 = panel.copy()
    p2.loc[p2["midx"] > cut, "units_sold"] = 0
    assert s1.global_idx == fc.SeasonalIndex.fit(p2, upto_idx=cut).global_idx
    # the real warehouse has a big December peak and a February trough
    assert max(s1.global_idx, key=s1.global_idx.get) == 12 and s1.get_global(12) > 1.4
    assert min(s1.global_idx, key=s1.global_idx.get) == 2 and s1.get_global(2) < 0.8


def test_seasonal_baseline_beats_naive_and_legacy_on_real_holdout(panel):
    """The headline claim of the backtest, asserted on the real data: 2024-H2 holdout."""
    start = 2024 * 12 + 6
    season = fc.SeasonalIndex.fit(panel, upto_idx=start - 1)
    f = fc.make_features(panel, season)
    f["t"] = f["midx"] + 1
    test = f[(f["t"] >= start) & f["target"].notna() & f["lag12_target"].notna()]
    base = fc.baseline_predictions(test)
    mae = lambda p: fc.metrics(test["target"], p)["mae"]  # noqa: E731
    assert mae(base["level_x_season"]) < mae(base["naive_last_month"]) * 0.85


def test_metrics_and_wape():
    m = fc.metrics([10, 20, 30], [12, 18, 33])
    assert round(m["mae"], 3) == 2.333 and round(m["wape"], 3) == round(7 / 60, 3) and round(m["bias"], 3) == 1.0


def test_inference_features_exist_for_every_product_without_target(panel):
    f = fc.make_features(panel, fc.SeasonalIndex.fit(panel), with_target=False)
    last = f[f["midx"] == panel["midx"].max()]
    assert len(last) == 500 and "target" not in f.columns
    assert last[fc.FEATURES].notna().all().all()


def test_trained_model_nonnegative_and_beats_naive_on_real_holdout(panel):
    start = 2024 * 12 + 6
    season = fc.SeasonalIndex.fit(panel, upto_idx=start - 1)
    f = fc.make_features(panel, season)
    f["t"] = f["midx"] + 1
    tr = f[(f["t"] < start) & f["target"].notna()]
    te = f[(f["t"] >= start) & f["target"].notna()]
    p = np.clip(fc.fit_xgb(tr).predict(te[fc.FEATURES]), 0, None)
    assert (p >= 0).all()
    assert fc.metrics(te["target"], p)["mae"] < fc.metrics(te["target"], te["lag1"])["mae"] * 0.85


# ------------------------------------------------------------------ anomaly detection
def test_real_series_shape(daily):
    assert len(daily) == 1096 and str(daily.index.min().date()) == "2022-01-01"


def test_december_peak_is_not_an_anomaly(daily):
    out = ac.detect_robust_z(daily)
    dec = out[out.index.month == 12]
    assert dec["is_anomaly"].mean() < 0.05            # legacy detector flagged mostly these days
    assert out["is_anomaly"].mean() < 0.02


def test_legacy_detector_does_flag_the_december_peak(daily):
    """Documents the bug being fixed: raw-value IsolationForest is dominated by seasonality."""
    flags = ac.detect_legacy_iforest(daily)
    assert (flags & (daily.index.month == 12)).sum() >= 0.5 * flags.sum()


def test_planted_anomalies_found_and_beats_legacy_on_real_series(daily):
    """Known anomalies planted into the real daily series, averaged over seeds."""
    zs, legacy = [], []
    for seed in range(10):
        mod, truth = ac.inject(daily, np.random.default_rng(seed))
        zs.append(ac.score_flags(ac.detect_robust_z(mod)["is_anomaly"], truth))
        legacy.append(ac.score_flags(ac.detect_legacy_iforest(mod), truth))
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))  # noqa: E731
    assert mean(zs, "precision") >= 0.7 and mean(zs, "recall") >= 0.65
    assert mean(zs, "f1") > mean(legacy, "f1") + 0.25


def test_zero_return_days_are_not_flagged_as_returns_anomalies(daily):
    zero = daily[daily["return_count"] == 0]
    out = ac.detect_robust_z(daily)
    flagged_zero = out.loc[zero.index]
    assert not ((flagged_zero["is_anomaly"]) & (flagged_zero["driver"] == "return_count")).any()


def test_returns_spike_is_flagged_and_attributed(daily):
    d = daily.index[300]
    mod = daily.copy()
    mod.loc[d, "return_count"] = daily["return_count"].max() * 4
    out = ac.detect_robust_z(mod)
    assert out.loc[d, "is_anomaly"] and out.loc[d, "driver"] == "return_count" and out.loc[d, "direction"] == "spike"


def test_injection_is_separated_and_labelled(daily):
    mod, truth = ac.inject(daily, np.random.default_rng(2))
    days = sorted(mod.index.get_loc(d) for d in truth)
    assert all(b - a >= 3 for a, b in zip(days, days[1:])) and set(truth.values()) == set(ac.INJECTION_TYPES)


def test_scoring_function_on_real_dates(daily):
    idx = daily.index[:5]
    flags = pd.Series([True, True, False, False, False], index=idx)
    s = ac.score_flags(flags, {idx[0]: "volume_spike_2.0x", idx[2]: "volume_dip_0.5x"})
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and s["precision"] == 0.5 and s["recall"] == 0.5
