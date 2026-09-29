"""
Demand forecasting: predicts next-month units_sold per product/category from
price_history + transactions aggregates. Time-based (not random) train/test split.
price_elasticity is excluded from the feature list per Section 4.1.
"""
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import read_sql, build_feature_frame, assert_no_forbidden_columns

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def load_data() -> pd.DataFrame:
    df = read_sql("""
        SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
               competitor_price_usd, price_index, is_promotional, units_sold, price_elasticity
        FROM fact_price_history
        ORDER BY product_id, year_month
    """)
    df["month_num"] = df.groupby("product_id").cumcount()
    df["lag_units_sold"] = df.groupby("product_id")["units_sold"].shift(1)
    df["target_next_units_sold"] = df.groupby("product_id")["units_sold"].shift(-1)
    df["is_promotional"] = df["is_promotional"].astype(int)
    df = df.dropna(subset=["lag_units_sold", "target_next_units_sold"])
    return df


def time_split(df: pd.DataFrame, holdout_months: int = 6):
    cutoff = df["year_month"].sort_values().unique()[-holdout_months]
    train = df[df["year_month"] < cutoff]
    test = df[df["year_month"] >= cutoff]
    return train, test


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_demand_forecasting")

    df = load_data()
    train, test = time_split(df)

    feature_cols = [
        "listed_price_usd", "base_price_usd", "competitor_price_usd",
        "price_index", "is_promotional", "month_num", "lag_units_sold",
    ]
    X_train = build_feature_frame(train[feature_cols])
    X_test = build_feature_frame(test[feature_cols])
    assert_no_forbidden_columns(X_train)
    y_train, y_test = train["target_next_units_sold"], test["target_next_units_sold"]

    with mlflow.start_run(run_name="xgb_demand_forecast"):
        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmspe = np.sqrt(np.mean(((y_test - preds) / y_test.replace(0, np.nan)) ** 2))

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "rmspe": float(rmspe)})
        mlflow.xgboost.log_model(model, "model")

        joblib.dump({"model": model, "features": feature_cols}, f"{MODEL_DIR}/demand_forecast.joblib")
        print(f"demand_forecast: MAE={mae:.2f} RMSE={rmse:.2f} RMSPE={rmspe:.3f}")


if __name__ == "__main__":
    main()
