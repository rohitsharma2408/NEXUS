"""
Anomaly detection over daily revenue/order-count/returns time series. Flags days that
look unusual so the ML Agent can surface them to the Investigation Agent.
"""
import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import IsolationForest

from features import read_sql

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def load_data() -> pd.DataFrame:
    sales = read_sql("""
        SELECT order_date, SUM(revenue_usd) AS revenue, COUNT(*) AS orders
        FROM fact_sales GROUP BY order_date ORDER BY order_date
    """)
    returns = read_sql("""
        SELECT return_date, COUNT(*) AS return_count
        FROM fact_returns GROUP BY return_date ORDER BY return_date
    """)
    df = sales.merge(returns, left_on="order_date", right_on="return_date", how="left")
    df["return_count"] = df["return_count"].fillna(0)
    df["dow"] = pd.to_datetime(df["order_date"]).dt.dayofweek
    return df[["order_date", "revenue", "orders", "return_count", "dow"]]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_anomaly_detection")

    df = load_data()
    features = ["revenue", "orders", "return_count", "dow"]
    X = df[features]

    with mlflow.start_run(run_name="isolation_forest_daily_revenue"):
        model = IsolationForest(n_estimators=300, contamination=0.03, random_state=42)
        model.fit(X)
        df["anomaly_score"] = model.decision_function(X)
        df["is_anomaly"] = model.predict(X) == -1

        n_anom = int(df["is_anomaly"].sum())
        mlflow.log_metric("n_anomalies_flagged", n_anom)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump({"model": model, "features": features}, f"{MODEL_DIR}/anomaly.joblib")
        df[df["is_anomaly"]].to_csv(f"{MODEL_DIR}/flagged_anomalies.csv", index=False)

        print(f"anomaly_detection: flagged {n_anom} / {len(df)} days")
        print(df[df["is_anomaly"]].sort_values("anomaly_score").head(10).to_string(index=False))


if __name__ == "__main__":
    main()
