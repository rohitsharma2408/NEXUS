"""
Anomaly detection over daily revenue / order-count / returns. Flags days that deviate from the
expected seasonal pattern so the ML Agent can surface them to the Investigation Agent.

Detector: robust z-score on seasonality-removed residuals (anomaly_core.py). Measured against
planted anomalies in evaluation/anomaly_eval.py (docs/anomaly_eval.md): F1 0.79 vs 0.39 for the
raw-value IsolationForest this script used before.
"""
import os
import joblib
import mlflow

import anomaly_core as ac
from features import read_sql

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def load_data():
    sales = read_sql("SELECT order_date, SUM(revenue_usd) AS revenue, COUNT(*) AS orders "
                     "FROM fact_sales GROUP BY order_date ORDER BY order_date")
    returns = read_sql("SELECT return_date AS order_date, COUNT(*) AS return_count "
                       "FROM fact_returns GROUP BY return_date")
    df = sales.merge(returns, on="order_date", how="left")
    df["return_count"] = df["return_count"].fillna(0)
    import pandas as pd
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df.set_index("order_date").sort_index()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_anomaly_detection")
    df = load_data()

    with mlflow.start_run(run_name="robust_z_residual_daily"):
        out = ac.detect_robust_z(df)
        flagged = out[out["is_anomaly"]].reset_index()
        n = len(flagged)
        mlflow.log_metrics({"n_anomalies_flagged": n, "z_threshold": ac.Z_THRESHOLD})
        joblib.dump({"detector": "robust_z_residual", "z_threshold": ac.Z_THRESHOLD},
                    f"{MODEL_DIR}/anomaly.joblib")
        cols = ["order_date", "revenue", "orders", "return_count", "anomaly_score", "driver", "direction"]
        flagged[cols].to_csv(f"{MODEL_DIR}/flagged_anomalies.csv", index=False)
        print(f"anomaly_detection: flagged {n} / {len(df)} days (|z| > {ac.Z_THRESHOLD})")
        print(flagged.sort_values("anomaly_score", ascending=False)[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
