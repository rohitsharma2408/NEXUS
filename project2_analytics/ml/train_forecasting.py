"""
Demand forecasting: predicts next-month units sold per product.

Production model = 50/50 blend of
  * level_x_season : trailing 12-month level x category seasonal index (a transparent baseline)
  * xgb_seasonal   : Poisson XGBoost on lags, rolling means, same-month-last-year and calendar
                     features (see forecast_core.py)
The blend, and every alternative, is measured by evaluation/forecast_backtest.py (walk-forward,
3 folds). The legacy model that used a 0..35 row counter as "month" scored worse than a
last-month-carried-forward guess; see docs/forecast_backtest.md.

Time-based holdout is kept here for the MLflow-logged metrics; price_elasticity is never used.
"""
import os
import joblib
import mlflow
import numpy as np

import forecast_core as fc
from features import read_sql, assert_no_forbidden_columns

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
HOLDOUT_MONTHS = 6


def load_panel():
    raw = read_sql("""
        SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
               competitor_price_usd, price_index, is_promotional, units_sold
        FROM fact_price_history ORDER BY product_id, year_month
    """)
    return fc.build_panel(raw)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_demand_forecasting")

    panel = load_panel()
    last_idx = int(panel["midx"].max())
    cutoff = last_idx - HOLDOUT_MONTHS + 1          # first holdout target month

    # holdout evaluation (seasonal index fit on pre-holdout months only)
    season_eval = fc.SeasonalIndex.fit(panel, upto_idx=cutoff - 1)
    feats = fc.make_features(panel, season_eval)
    feats["target_idx"] = feats["midx"] + 1
    tr = feats[(feats["target_idx"] < cutoff) & feats["target"].notna()]
    te = feats[(feats["target_idx"] >= cutoff) & feats["target"].notna()]
    assert_no_forbidden_columns(tr[fc.FEATURES])

    with mlflow.start_run(run_name="seasonal_blend_forecast"):
        m_eval = fc.fit_xgb(tr)
        blend = 0.5 * te["level_x_season"].to_numpy() + 0.5 * np.clip(m_eval.predict(te[fc.FEATURES]), 0, None)
        res = fc.metrics(te["target"], blend)
        naive = fc.metrics(te["target"], te["lag1"])
        res["skill_vs_naive_pct"] = round((1 - res["mae"] / naive["mae"]) * 100, 1)
        mlflow.log_metrics({"mae": res["mae"], "rmse": res["rmse"], "wape": res["wape"],
                            "naive_mae": naive["mae"], "skill_vs_naive_pct": res["skill_vs_naive_pct"]})

        # final model: refit on everything available, seasonal index from all history
        season = fc.SeasonalIndex.fit(panel)
        full = fc.make_features(panel, season)
        full = full[full["target"].notna()]
        model = fc.fit_xgb(full)
        joblib.dump({
            "model": model, "features": fc.FEATURES, "season": season,
            "categories": sorted(panel["category"].unique()), "blend_weight_seasonal": 0.5,
            "holdout_metrics": res, "naive_holdout_mae": naive["mae"],
            "last_month": fc.idx_to_ym(last_idx),
        }, f"{MODEL_DIR}/demand_forecast.joblib")
        print(f"demand_forecast (blend): holdout MAE={res['mae']:.2f} RMSE={res['rmse']:.2f} "
              f"WAPE={res['wape']:.3f} | naive MAE={naive['mae']:.2f} "
              f"| skill vs naive {res['skill_vs_naive_pct']:+.1f}%")


if __name__ == "__main__":
    main()
