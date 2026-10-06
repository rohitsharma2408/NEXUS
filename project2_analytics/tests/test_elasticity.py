"""Elasticity validation, run on the REAL warehouse (fact_price_history). Skipped automatically
when the warehouse is not reachable. No synthetic data.

These check the estimator's behaviour on the real data; they deliberately do not assert what the
verdict is, because that is the finding, not a requirement."""
import numpy as np
import pytest

from features import read_sql
from validate_elasticity import MIN_MOVES, estimate_elasticity, summarize, verdict


@pytest.fixture(scope="module")
def real():
    try:
        df = read_sql("SELECT product_id, year_month, listed_price_usd, units_sold, price_elasticity "
                      "FROM fact_price_history ORDER BY product_id, year_month")
    except Exception as e:
        pytest.skip(f"warehouse not available: {e}")
    return df, estimate_elasticity(df)


def test_estimates_exist_and_are_well_formed(real):
    df, res = real
    assert len(res) > 0, "no product has enough real price moves to estimate"
    assert res["derived_elasticity"].notna().all() and np.isfinite(res["se"]).all()
    assert (res["n_price_moves"] >= MIN_MOVES).all()
    assert res["product_id"].is_unique


def test_estimates_are_not_the_old_exploding_ratio(real):
    # the old estimator gave |derived| around 5-11 from tiny price moves
    _, res = real
    assert res["derived_elasticity"].abs().median() < 5


def test_static_value_is_used_only_for_comparison(real):
    # same input with the static column scrambled must give identical derived estimates
    df, res = real
    scrambled = df.copy()
    scrambled["price_elasticity"] = np.random.default_rng(0).permutation(scrambled["price_elasticity"].to_numpy())
    res2 = estimate_elasticity(scrambled)
    a = res.set_index("product_id")["derived_elasticity"].sort_index()
    b = res2.set_index("product_id")["derived_elasticity"].sort_index()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_summary_and_verdict_run_on_real_data(real):
    df, res = real
    assert verdict(summarize(res, df["product_id"].nunique())).split(" ")[0] in {
        "consistent", "systematic", "inconclusive"}
