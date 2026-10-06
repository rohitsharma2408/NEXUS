"""
Walk-forward backtest for the demand forecaster.

For each fold the models are fit ONLY on origins whose target month is before the fold's test
window, and the seasonal index is estimated from that same window. Test rows are 1-step-ahead
forecasts (predict month t+1 from data up to t), exactly how the ML agent uses the model.

    python evaluation/forecast_backtest.py                 # needs DATABASE_URL
    python evaluation/forecast_backtest.py --csv panel.csv # or a CSV of fact_price_history

Writes evaluation/forecast_backtest_results.json and docs/forecast_backtest.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project2_analytics" / "ml"))
import forecast_core as fc  # noqa: E402

# (label, first test target month, last test target month)
FOLDS = [
    ("2023-H2", "2023-07", "2023-12"),
    ("2024-H1", "2024-01", "2024-06"),
    ("2024-H2", "2024-07", "2024-12"),
]


def load_panel(csv: str | None) -> pd.DataFrame:
    if csv:
        raw = pd.read_csv(csv)
    else:
        from features import read_sql
        raw = fc.build_panel  # placeholder to keep linters quiet
        raw = read_sql("""SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
                                 competitor_price_usd, price_index, is_promotional, units_sold
                          FROM fact_price_history ORDER BY product_id, year_month""")
    return fc.build_panel(raw)


def run_fold(panel: pd.DataFrame, label: str, start_ym: str, end_ym: str) -> dict:
    s_idx = int(fc.ym_to_idx(pd.Series([start_ym]))[0])
    e_idx = int(fc.ym_to_idx(pd.Series([end_ym]))[0])

    season = fc.SeasonalIndex.fit(panel, upto_idx=s_idx - 1)
    feats = fc.make_features(panel, season)
    feats["target_idx"] = feats["midx"] + 1
    train = feats[(feats["target_idx"] < s_idx) & feats["target"].notna()]
    test = feats[(feats["target_idx"] >= s_idx) & (feats["target_idx"] <= e_idx) & feats["target"].notna()]
    # a fair comparison needs every baseline to be defined on the same rows
    test = test[test["lag12_target"].notna()].reset_index(drop=True)

    preds = fc.baseline_predictions(test)

    model = fc.fit_xgb(train)
    preds["xgb_seasonal"] = np.clip(model.predict(test[fc.FEATURES]), 0, None)

    preds["blend_seasonal_xgb"] = 0.5 * preds["level_x_season"] + 0.5 * preds["xgb_seasonal"]

    # ---- legacy model, rebuilt exactly as the old train_forecasting.py did, on the same rows
    from xgboost import XGBRegressor
    legacy = fc.make_legacy_features(panel)
    legacy["target_idx"] = legacy["midx"] + 1
    ltr = legacy[legacy["target_idx"] < s_idx]
    lte = legacy[(legacy["target_idx"] >= s_idx) & (legacy["target_idx"] <= e_idx)]
    lm = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    lm.fit(ltr[fc.LEGACY_FEATURES], ltr["target"])
    lte = lte.assign(pred=lm.predict(lte[fc.LEGACY_FEATURES]))
    key = ["product_id", "target_idx"]
    joined = test.merge(lte[key + ["pred"]], on=key, how="left")
    preds["legacy_xgb"] = joined["pred"].to_numpy()

    y = test["target"].to_numpy()
    rows = {}
    for name, p in preds.items():
        m = fc.metrics(y, p)
        m["category_month_wape"] = fc.aggregate_wape(test, np.asarray(p, float))
        rows[name] = m
    return {"fold": label, "test_rows": int(len(test)), "train_rows": int(len(train)), "models": rows}


def summarize(folds: list[dict]) -> dict:
    names = list(folds[0]["models"].keys())
    avg = {}
    for n in names:
        avg[n] = {k: float(np.mean([f["models"][n][k] for f in folds]))
                  for k in ("mae", "rmse", "wape", "category_month_wape", "bias")}
    best = min(avg, key=lambda k: avg[k]["mae"])
    base = avg["naive_last_month"]["mae"]
    legacy = avg["legacy_xgb"]["mae"]
    return {
        "mean_over_folds": avg,
        "best_by_mae": best,
        "skill_vs_naive_pct": {n: round((1 - avg[n]["mae"] / base) * 100, 1) for n in names},
        "improvement_vs_legacy_pct": round((1 - avg["xgb_seasonal"]["mae"] / legacy) * 100, 1),
    }


def to_markdown(res: dict) -> str:
    s = res["summary"]
    lines = ["# Demand forecast backtest", "",
             "Walk-forward, 1-step-ahead (predict next month's units per product). Lower is better.",
             "Each fold trains only on months before its test window; the seasonal index is also",
             "estimated from that window only.", ""]
    lines += ["| Model | MAE | RMSE | WAPE | Category-month WAPE | Skill vs naive |", "|---|---|---|---|---|---|"]
    for n, m in sorted(s["mean_over_folds"].items(), key=lambda kv: kv[1]["mae"]):
        lines.append(f"| {n} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['wape']*100:.1f}% | "
                     f"{m['category_month_wape']*100:.1f}% | {s['skill_vs_naive_pct'][n]:+.1f}% |")
    lines += ["", f"Best by MAE: **{s['best_by_mae']}**. "
              f"xgb_seasonal vs the previous model: **{s['improvement_vs_legacy_pct']:+.1f}% MAE**.", "",
              "## Per fold (MAE)", "", "| Fold | " + " | ".join(res["folds"][0]["models"].keys()) + " |",
              "|---|" + "---|" * len(res["folds"][0]["models"])]
    for f in res["folds"]:
        lines.append(f"| {f['fold']} | " + " | ".join(f"{m['mae']:.2f}" for m in f["models"].values()) + " |")
    lines += ["", "Product-month sales are noisy counts (mean about 13 units, lag-1 autocorrelation about 0.05),",
              "so a large part of the error is irreducible at the single-product level. The",
              "category-month column shows the error after products are summed, which is the level",
              "planning decisions are made at."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out-json", default=str(ROOT / "evaluation" / "forecast_backtest_results.json"))
    ap.add_argument("--out-md", default=str(ROOT / "docs" / "forecast_backtest.md"))
    a = ap.parse_args()

    panel = load_panel(a.csv)
    folds = [run_fold(panel, *f) for f in FOLDS]
    res = {"folds": folds, "summary": summarize(folds)}
    Path(a.out_json).write_text(json.dumps(res, indent=2))
    md = to_markdown(res)
    Path(a.out_md).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
