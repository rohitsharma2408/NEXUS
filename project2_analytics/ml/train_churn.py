"""
Customer churn model. "Churned" = no order in the most recent 90 days relative to the
dataset's max order_date, among customers registered before that window.

gender/age/country are pulled in only to compute the low_confidence flag, then
stripped by build_feature_frame() before .fit() — see features.py.
"""
import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report

from features import read_sql, build_feature_frame, flag_low_confidence, assert_no_forbidden_columns

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def load_data() -> pd.DataFrame:
    customers = read_sql("SELECT * FROM dim_customer")
    sales = read_sql("""
        SELECT customer_id, order_date, revenue_usd, transaction_id
        FROM fact_sales
    """)
    sales["order_date"] = pd.to_datetime(sales["order_date"])
    max_date = sales["order_date"].max()
    window_start = max_date - pd.Timedelta(days=90)

    agg = sales.groupby("customer_id").agg(
        total_orders=("transaction_id", "count"),
        total_revenue=("revenue_usd", "sum"),
        last_order_date=("order_date", "max"),
    ).reset_index()

    df = customers.merge(agg, on="customer_id", how="left")
    df["total_orders"] = df["total_orders"].fillna(0)
    df["total_revenue"] = df["total_revenue"].fillna(0)
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["days_since_last_order"] = (max_date - df["last_order_date"]).dt.days.fillna(999)
    df["tenure_days"] = (max_date - df["registration_date"]).dt.days
    df["churned"] = (df["days_since_last_order"] > 90).astype(int)

    # eligible only: registered long enough ago to have had a chance to reorder
    df = df[df["tenure_days"] > 90].copy()

    # low_confidence flag computed from raw customer table (has gender/country), before stripping
    df = flag_low_confidence(df, customers)
    return df


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_churn")

    df = load_data()
    feature_cols = ["total_orders", "total_revenue", "tenure_days", "is_premium"]
    X = build_feature_frame(df[feature_cols + ["customer_id"]].drop(columns=["customer_id"]))
    assert_no_forbidden_columns(X)
    y = df["churned"]

    X_train, X_test, y_train, y_test, lc_train, lc_test = train_test_split(
        X, y, df["low_confidence"], test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="rf_churn"):
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, preds)

        mlflow.log_metrics({"roc_auc": auc, "f1": f1})
        mlflow.sklearn.log_model(model, "model")

        joblib.dump({"model": model, "features": feature_cols}, f"{MODEL_DIR}/churn.joblib")

        print(f"churn: ROC-AUC={auc:.3f} F1={f1:.3f}")
        print(f"low_confidence test rows: {lc_test.sum()} / {len(lc_test)}")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
