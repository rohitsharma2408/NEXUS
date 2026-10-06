"""
Anomaly-detection evaluation (labelled injection protocol).

    python evaluation/anomaly_eval.py            # needs DATABASE_URL, or --csv daily.csv

For each of N seeds: inject ~3% known anomalies (volume spikes/dips, returns spikes) into the
real daily series, run every detector, score precision / recall / F1 against the injected days.
Also reports each detector's flags on the CLEAN real series (what it would tell you today).
Writes evaluation/anomaly_eval_results.json and docs/anomaly_eval.md.
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
import anomaly_core as ac  # noqa: E402


def load_daily(csv: str | None) -> pd.DataFrame:
    if csv:
        d = pd.read_csv(csv, parse_dates=["order_date"])
    else:
        from features import read_sql
        sales = read_sql("SELECT order_date, SUM(revenue_usd) AS revenue, COUNT(*) AS orders "
                         "FROM fact_sales GROUP BY order_date ORDER BY order_date")
        ret = read_sql("SELECT return_date AS order_date, COUNT(*) AS return_count "
                       "FROM fact_returns GROUP BY return_date")
        d = sales.merge(ret, on="order_date", how="left")
        d["order_date"] = pd.to_datetime(d["order_date"])
    d["return_count"] = d["return_count"].fillna(0)
    return d.set_index("order_date").sort_index()[["revenue", "orders", "return_count"]]


DETECTORS = {
    "legacy_isolation_forest": lambda df: ac.detect_legacy_iforest(df),
    "isolation_forest_residual": lambda df: ac.detect_iforest_residual(df),
    "robust_z_residual": lambda df: ac.detect_robust_z(df)["is_anomaly"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()

    df = load_daily(a.csv)
    agg = {n: [] for n in DETECTORS}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        mod, truth = ac.inject(df, rng)
        for n, fn in DETECTORS.items():
            agg[n].append(ac.score_flags(fn(mod), truth))

    summary = {}
    for n, runs in agg.items():
        summary[n] = {
            "precision": float(np.mean([r["precision"] for r in runs])),
            "recall": float(np.mean([r["recall"] for r in runs])),
            "f1": float(np.mean([r["f1"] for r in runs])),
            "avg_flagged": float(np.mean([r["flagged"] for r in runs])),
            "recall_by_type": {t: float(np.nanmean([r["recall_by_type"][t] for r in runs]))
                               for t in ac.INJECTION_TYPES},
        }
    clean = {n: int(fn(df).sum()) for n, fn in DETECTORS.items()}
    res = {"days": len(df), "seeds": a.seeds, "injected_per_run": len(truth), "summary": summary,
           "flags_on_clean_real_data": clean}
    Path(ROOT / "evaluation" / "anomaly_eval_results.json").write_text(json.dumps(res, indent=2))

    L = ["# Anomaly detection evaluation", "",
         f"Injection protocol: {res['injected_per_run']} known anomalies planted in the real "
         f"{res['days']}-day series per run, {a.seeds} random seeds, averaged. The warehouse has no "
         "real labels, so this measures detection power for the stated magnitudes, not performance "
         "on unknown real incidents.", "",
         "| Detector | Precision | Recall | F1 | Days flagged / run | Flags on clean data |", "|---|---|---|---|---|---|"]
    for n, s in sorted(summary.items(), key=lambda kv: -kv[1]["f1"]):
        L.append(f"| {n} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | "
                 f"{s['avg_flagged']:.0f} | {clean[n]} |")
    L += ["", "## Recall by anomaly type", "", "| Detector | " + " | ".join(ac.INJECTION_TYPES) + " |",
          "|---|" + "---|" * len(ac.INJECTION_TYPES)]
    for n, s in summary.items():
        L.append(f"| {n} | " + " | ".join(f"{s['recall_by_type'][t]:.2f}" for t in ac.INJECTION_TYPES) + " |")
    md = "\n".join(L)
    Path(ROOT / "docs" / "anomaly_eval.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
