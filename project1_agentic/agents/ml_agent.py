"""ML Agent — invokes the models trained in project2_analytics/ml/train_*.py."""
import sys
from pathlib import Path
from dataclasses import dataclass

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project2_analytics" / "ml"))
from features import read_sql  # noqa: E402
from config import MODEL_DIR  # noqa: E402


@dataclass
class MLResult:
    model_name: str
    summary: dict


def _load(name: str):
    path = Path(MODEL_DIR) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train_all.py first.")
    return joblib.load(path)


def forecast_next_month(product_id: str | None = None) -> MLResult:
    bundle = _load("demand_forecast")
    model, feats = bundle["model"], bundle["features"]
    query = "SELECT * FROM fact_price_history"
    if product_id:
        query += f" WHERE product_id = '{product_id}'"
    df = read_sql(query + " ORDER BY product_id, year_month")
    latest = df.sort_values("year_month").groupby("product_id").tail(1).copy()
    latest["month_num"] = 1
    latest["lag_units_sold"] = latest["units_sold"]
    latest["is_promotional"] = latest["is_promotional"].astype(int)
    X = latest[feats]
    latest["predicted_next_month_units"] = model.predict(X)
    top = latest.sort_values("predicted_next_month_units", ascending=False).head(10)
    return MLResult("demand_forecast", {"predictions": top[["product_id", "predicted_next_month_units"]].to_dict("records")})


def get_recent_anomalies() -> MLResult:
    path = Path(MODEL_DIR) / "flagged_anomalies.csv"
    if not path.exists():
        return MLResult("anomaly", {"anomalies": [], "note": "No anomaly model output found; run train_anomaly.py."})
    df = pd.read_csv(path).sort_values("order_date", ascending=False).head(10)
    return MLResult("anomaly", {"anomalies": df.to_dict("records")})


def get_churn_risk(top_n: int = 10) -> MLResult:
    bundle = _load("churn")
    model, feats = bundle["model"], bundle["features"]
    customers = read_sql("SELECT * FROM dim_customer")
    sales = read_sql("SELECT customer_id, order_date, revenue_usd, transaction_id FROM fact_sales")
    max_date = sales["order_date"].max()
    agg = sales.groupby("customer_id").agg(
        total_orders=("transaction_id", "count"), total_revenue=("revenue_usd", "sum")
    ).reset_index()
    df = customers.merge(agg, on="customer_id", how="left").fillna(0)
    df["tenure_days"] = (max_date - df["registration_date"]).dt.days
    X = df[feats]
    df["churn_probability"] = model.predict_proba(X)[:, 1]
    top = df.sort_values("churn_probability", ascending=False).head(top_n)
    return MLResult("churn", {"at_risk_customers": top[["customer_id", "churn_probability"]].to_dict("records")})


def get_supplier_risk() -> MLResult:
    df = read_sql("SELECT * FROM kpi_supplier_risk WHERE supplier_risk_level = 'HIGH' LIMIT 20")
    return MLResult("supplier_risk", {"high_risk_suppliers": df.to_dict("records")})
