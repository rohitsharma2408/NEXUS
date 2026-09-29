"""
Supplier risk model: predicts a composite risk label (LOW/MEDIUM/HIGH) from
reliability_score, lead_time_days, MOQ, and current stock position, so the ML Agent
can proactively flag reorder risk rather than the SQL Agent's static kpi_supplier_risk
view alone.
"""
import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from features import read_sql, build_feature_frame, assert_no_forbidden_columns

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def load_data() -> pd.DataFrame:
    df = read_sql("""
        SELECT sc.product_id, sc.reliability_score, sc.lead_time_days, sc.min_order_qty,
               sc.unit_cost_usd, sc.holding_cost_pct, i.stock_units, i.reorder_point
        FROM fact_supplier_costs sc
        LEFT JOIN fact_inventory i ON sc.product_id = i.product_id
        WHERE sc.is_primary = true
    """).dropna()

    def label(row):
        if row["reliability_score"] < 0.7 and row["stock_units"] <= row["reorder_point"]:
            return "HIGH"
        if row["reliability_score"] < 0.85:
            return "MEDIUM"
        return "LOW"

    df["risk_label"] = df.apply(label, axis=1)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_supplier_risk")

    df = load_data()
    feature_cols = ["lead_time_days", "min_order_qty", "unit_cost_usd", "holding_cost_pct",
                     "stock_units", "reorder_point"]
    # reliability_score is the strongest signal but is also literally how we built the
    # label above; keep it out of X to avoid trivially perfect (leaked) accuracy, and
    # let the model instead learn the *operational* correlates of risk.
    X = build_feature_frame(df[feature_cols])
    assert_no_forbidden_columns(X)
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="gbc_supplier_risk"):
        model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump({"model": model, "features": feature_cols}, f"{MODEL_DIR}/supplier_risk.joblib")
        print(f"supplier_risk: F1-macro={f1:.3f}")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
