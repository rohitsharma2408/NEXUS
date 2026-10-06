"""ML Agent — invokes the models trained in project2_analytics/ml/train_*.py."""
import sys
from pathlib import Path
from dataclasses import dataclass

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "project2_analytics" / "ml"))
from features import read_sql  # noqa: E402
import forecast_core  # noqa: E402,F401  (needed so joblib can unpickle SeasonalIndex)
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
    """Next-month UNITS forecast (not revenue). Same feature builder as training, so there is
    no train/serve skew. Returns a total, a per-category split, top products, and the model's
    measured holdout accuracy so the analyst can state how much to trust it."""
    import numpy as np
    import forecast_core as fc

    bundle = _load("demand_forecast")
    model, season, cats = bundle["model"], bundle["season"], bundle["categories"]
    w = bundle.get("blend_weight_seasonal", 0.5)
    raw = read_sql(
        "SELECT product_id, category, year_month, listed_price_usd, base_price_usd, "
        "competitor_price_usd, price_index, is_promotional, units_sold "
        "FROM fact_price_history ORDER BY product_id, year_month"
    )
    panel = fc.build_panel(raw)
    feats = fc.make_features(panel, season, categories=cats, with_target=False)
    latest = feats.sort_values("midx").groupby("product_id").tail(1).copy()
    if product_id:
        latest = latest[latest["product_id"] == product_id]
    pred_xgb = np.clip(model.predict(latest[bundle["features"]]), 0, None)
    latest["predicted_next_month_units"] = (w * latest["level_x_season"].to_numpy()
                                            + (1 - w) * pred_xgb).round(1)
    target_ym = fc.idx_to_ym(int(latest["midx"].max()) + 1)
    by_cat = (latest.groupby("category")["predicted_next_month_units"].sum().round(0)
              .sort_values(ascending=False))
    top = latest.sort_values("predicted_next_month_units", ascending=False).head(10)
    hm = bundle.get("holdout_metrics", {})
    return MLResult("demand_forecast", {
        "forecast_month": target_ym,
        "unit": "units sold (not revenue)",
        "total_predicted_units": round(float(latest["predicted_next_month_units"].sum()), 0),
        "predicted_units_by_category": by_cat.to_dict(),
        "predictions": top[["product_id", "predicted_next_month_units"]].to_dict("records"),
        "accuracy_note": (f"Holdout MAE {hm.get('mae', float('nan')):.2f} units per product-month, "
                          f"{hm.get('skill_vs_naive_pct', 0):+.1f}% vs a last-month guess; "
                          "category totals are far more reliable than single products."),
    })


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
