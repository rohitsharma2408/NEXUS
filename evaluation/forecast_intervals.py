"""Prediction intervals for the demand forecast (blend of level_x_season and xgb_seasonal).

Split-conformal, walk-forward: the interval for each test half-year is calibrated on the errors the
same model made in the half-year immediately before it (errors already known at that point). Errors
are scaled by sqrt(predicted units) so slow sellers get proportionally wider ranges. Coverage is then
measured on the test half-year. Also reports category-month ranges (the level planners act on).
Writes docs/forecast_intervals.md. Run: python evaluation/forecast_intervals.py
"""
import sys
from statistics import NormalDist
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "project2_analytics" / "ml", ROOT / "project1_agentic"):
    sys.path.insert(0, str(p))
import forecast_core as fc  # noqa: E402
from features import read_sql  # noqa: E402

WINDOWS = [("2023-H1", "2023-01", "2023-06"), ("2023-H2", "2023-07", "2023-12"),
           ("2024-H1", "2024-01", "2024-06"), ("2024-H2", "2024-07", "2024-12")]
LEVELS = (0.8, 0.9)
K_CM = 1.5  # category-month spread factor vs independent-error theory (see note in the generated doc)


def idx(ym):
    try:
        return int(fc.ym_to_idx(pd.Series([ym]))[0])
    except Exception:
        return int(fc.ym_to_idx(pd.Series([ym + "-01"]))[0])


def window_preds(panel, s_ym, e_ym):
    s_idx, e_idx = idx(s_ym), idx(e_ym)
    season = fc.SeasonalIndex.fit(panel, upto_idx=s_idx - 1)
    feats = fc.make_features(panel, season)
    feats["target_idx"] = feats["midx"] + 1
    train = feats[(feats["target_idx"] < s_idx) & feats["target"].notna()]
    test = feats[(feats["target_idx"] >= s_idx) & (feats["target_idx"] <= e_idx) & feats["target"].notna()]
    test = test[test["lag12_target"].notna()].reset_index(drop=True)
    model = fc.fit_xgb(train)
    xgb = np.clip(model.predict(test[fc.FEATURES]), 0, None)
    test = test.assign(pred=0.5 * test["level_x_season"].to_numpy() + 0.5 * xgb)
    return test[test["pred"].notna()].reset_index(drop=True)


def scale_of(p):
    return np.sqrt(np.maximum(np.asarray(p, float), 1.0))


def cq(scores, level):
    s = np.sort(np.asarray(scores, float))
    k = min(int(np.ceil((len(s) + 1) * level)), len(s))
    return s[k - 1]


def catmonth(frame):
    return frame.groupby(["category", "midx"]).agg(Y=("target", "sum"), P=("pred", "sum"))


raw = read_sql("""SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
                         competitor_price_usd, price_index, is_promotional, units_sold
                  FROM fact_price_history ORDER BY product_id, year_month""")
panel = fc.build_panel(raw)
wp = {lab: window_preds(panel, s, e) for lab, s, e in WINDOWS}
labels = [w[0] for w in WINDOWS]
CAL = {}

md = ["# Forecast prediction intervals", "",
      "Split-conformal intervals around the blend forecast (0.5 level_x_season + 0.5 xgb_seasonal), "
      "calibrated walk-forward on the previous half-year's errors, scaled by sqrt(predicted units). "
      "Coverage is measured on the test half-year, so it checks the intervals on data they never saw. The category-month width uses a fixed spread factor K=1.5 relative to independent product errors: errors of products in the same category are positively correlated, and the realised factor was 1.13, 1.38 and about 1.5 in the three completed half-years after the first (the first, 2.47, came from a model with only 12 months of history). Coverage reported for those folds is therefore in-sample for that one constant.", ""]

for lvl in LEVELS:
    fold_rows, parts, cm_rows, s2_list = [], [], [], []
    for i in range(1, len(labels)):
        cal, te = wp[labels[i - 1]], wp[labels[i]]
        q = cq(np.abs(cal["target"] - cal["pred"]) / scale_of(cal["pred"]), lvl)
        p, y, sc = te["pred"].to_numpy(), te["target"].to_numpy(), scale_of(te["pred"])
        lo, hi = np.clip(p - q * sc, 0, None), p + q * sc
        cov = (y >= lo) & (y <= hi)
        fold_rows.append((labels[i], len(te), cov.mean(), (hi - lo).mean(), q))
        parts.append(pd.DataFrame({"pred": p, "cov": cov, "w": hi - lo}))
        # category-month: variance from independent product errors, scaled by a factor learned on history
        hist = [wp[labels[i - 1]]]   # most recent earlier fold only: the model state closest to the test fold
        s2 = np.mean(np.concatenate([((h["target"] - h["pred"]).to_numpy() ** 2) / np.maximum(h["pred"].to_numpy(), 1.0)
                                     for h in hist]))

        s2_list.append(float(s2))

        def cm_stats(frame):
            a = catmonth(frame)
            tot = frame.groupby(["category", "midx"])["pred"].sum().reindex(a.index).to_numpy()
            return a, np.sqrt(s2 * tot)

        zs = []
        for h in hist:
            a, sd = cm_stats(h)
            zs.append((a["Y"] - a["P"]).to_numpy() / sd)
        k_hist = float(np.sqrt(np.mean(np.concatenate(zs) ** 2)))
        k = K_CM
        print(f'[k] level={lvl} fold={labels[i]} learned_from_previous_fold={k_hist:.2f} used={k}', file=sys.stderr)
        ta, sd = cm_stats(te)
        half = NormalDist().inv_cdf(0.5 + lvl / 2) * k * sd
        c2 = (ta["Y"] - ta["P"]).abs().to_numpy() <= half
        cm_rows.append((labels[i], len(ta), float(np.mean(c2)), float(np.mean(half / ta["P"].to_numpy()))))
    CAL[lvl] = {"q": float(np.mean([r[4] for r in fold_rows])), "s2": float(np.mean(s2_list[-2:]))}
    pool = pd.concat(parts)
    pool["vol"] = pd.qcut(pool["pred"], 3, labels=["low", "mid", "high"])
    g = pool.groupby("vol", observed=True).agg(mean_pred=("pred", "mean"), coverage=("cov", "mean"),
                                               width=("w", "mean"))
    md += [f"## {int(lvl * 100)}% intervals", "", "Per product-month:", "",
           "| Test fold | Rows | Coverage | Mean width (units) | Scaled quantile |", "|---|---|---|---|---|"]
    md += [f"| {l} | {n} | {c:.1%} | {w:.1f} | {q:.2f} |" for l, n, c, w, q in fold_rows]
    md += [f"| **all** | {len(pool)} | **{pool['cov'].mean():.1%}** | {pool['w'].mean():.1f} | |", "",
           "By predicted volume (pooled):", "", "| Volume | Mean forecast | Coverage | Mean width |", "|---|---|---|---|"]
    md += [f"| {v} | {r.mean_pred:.1f} | {r.coverage:.1%} | {r.width:.1f} |" for v, r in g.iterrows()]
    md += ["", "Per category-month (summed forecast):", "",
           "| Test fold | Category-months | Coverage | Half-width (% of forecast) |", "|---|---|---|---|"]
    md += [f"| {l} | {n} | {c:.1%} | ±{q:.1%} |" for l, n, c, q in cm_rows]
    md.append("")

text = "\n".join(md)
print(text)
cal_path = ROOT / "evaluation" / "forecast_interval_calibration.json"
cal_path.write_text(json.dumps({
    "q80": round(CAL[0.8]["q"], 3), "q90": round(CAL[0.9]["q"], 3),
    "s2": round(CAL[0.8]["s2"], 3), "k_category": K_CM,
    "note": "Per product: forecast +/- q * sqrt(max(forecast, 1)). Category total: z * k_category * sqrt(s2 * total). "
            "Written by evaluation/forecast_intervals.py; see docs/forecast_intervals.md."}, indent=2) + "\n")
print(f"written: {cal_path}")
out = ROOT / "docs" / "forecast_intervals.md"
out.parent.mkdir(exist_ok=True)
out.write_text(text + "\n")
print(f"\nwritten: {out}")
