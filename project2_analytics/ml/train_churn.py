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

    # Time-based split (not random) per the blueprint's Section 9 requirement: the
    # earliest-registered 80% of eligible customers train the model, the most-recently-
    # registered 20% test it, so the model is evaluated the way it would actually be used —
    # predicting churn for customers it hasn't seen the full history of yet.
    # Split by a DATE THRESHOLD, not a row-count index: with real registration dates,
    # multiple customers share the same day, so an index-based cut can split same-day
    # customers across both train and test. A date threshold keeps every same-day
    # customer together on one side, so there's no ambiguity at the boundary.
    df_sorted = df.sort_values("registration_date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * 0.8)
    boundary_date = df_sorted.iloc[split_idx]["registration_date"]
    train_df = df_sorted[df_sorted["registration_date"] < boundary_date]
    test_df = df_sorted[df_sorted["registration_date"] >= boundary_date]

    X_train = build_feature_frame(train_df[feature_cols])
    X_test = build_feature_frame(test_df[feature_cols])
    y_train, y_test = train_df["churned"], test_df["churned"]
    lc_test = test_df["low_confidence"]

    with mlflow.start_run(run_name="rf_churn"):
        model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
        assert_no_forbidden_columns(X_train)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, preds)

        mlflow.log_metrics({"roc_auc": auc, "f1": f1})
        joblib.dump({"model": model, "features": feature_cols}, f"{MODEL_DIR}/churn.joblib")  # the artifact the API loads
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:   # newer MLflow refuses to serialise sklearn trees; the joblib above is what matters
            print(f"[mlflow] model artifact not logged ({type(e).__name__}); metrics were logged")

        joblib.dump({"model": model, "features": feature_cols}, f"{MODEL_DIR}/churn.joblib")

        print(f"churn: ROC-AUC={auc:.3f} F1={f1:.3f}")
        print(f"low_confidence test rows: {lc_test.sum()} / {len(lc_test)}")
        print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
