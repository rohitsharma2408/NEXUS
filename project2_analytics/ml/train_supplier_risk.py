"""
Supplier risk — NOT a trained classifier.

The original version of this script trained a GradientBoostingClassifier on a label
built from reliability_score and stock position, then fed reliability_score and stock
position back in as features: the model was trivially predicting its own label-
construction rule, reported meaningless "accuracy," and still never learned the HIGH
class (it never appeared with enough examples in the random split). There is no real
observed outcome (an actual stockout, an actual supplier failure) in this dataset to
train against — just a rule.

So: kpi_supplier_risk (project2_analytics/sql/kpi_views.sql) IS the supplier risk model,
as a transparent, auditable SQL rule rather than an opaque classifier trained on its own
label. project1_agentic/agents/ml_agent.py:get_supplier_risk() already queries that view
directly. This script only validates the rule's output distribution so a human can sanity
-check it (e.g. "is HIGH actually rare, as intended, or did the thresholds misfire").

If real outcome data (actual late deliveries, actual stockouts with a timestamp) is ever
added to the warehouse, THAT would be a legitimate classification target and a real model
would belong here instead.
"""
import os
import mlflow
import pandas as pd

from features import read_sql

MODEL_DIR = os.environ.get("MODEL_DIR", "models")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment("nexus_supplier_risk")

    df = read_sql("SELECT * FROM kpi_supplier_risk")

    with mlflow.start_run(run_name="supplier_risk_rule_validation"):
        counts = df["supplier_risk_level"].value_counts()
        for level in ["LOW", "MEDIUM", "HIGH"]:
            mlflow.log_metric(f"n_{level.lower()}", int(counts.get(level, 0)))

        df.to_csv(f"{MODEL_DIR}/supplier_risk_snapshot.csv", index=False)

        print("supplier_risk: rule-based distribution (from kpi_supplier_risk view)")
        print(counts.to_string())
        n_high = int(counts.get("HIGH", 0))
        low_rel = df["reliability_score"] < 0.80
        at_reorder = df["stock_units"] <= df["reorder_point"]
        no_stock_row = int(df["stock_units"].isna().sum())
        print("\nWhy the split looks like this (HIGH needs BOTH conditions below):")
        print(f"  reliability < 0.80:           {int(low_rel.sum())}")
        print(f"  stock <= reorder point:       {int(at_reorder.sum())}")
        print(f"  both (= HIGH):                {int((low_rel & at_reorder).sum())}")
        print(f"  no inventory row (never HIGH): {no_stock_row}")
        print("  reliability_score quantiles:  " + ", ".join(
            f"p{int(q*100)}={df['reliability_score'].quantile(q):.2f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)))
        for k, v in {"n_rel_lt_070": int(low_rel.sum()), "n_at_reorder": int(at_reorder.sum()),
                     "n_no_inventory_row": no_stock_row}.items():
            mlflow.log_metric(k, v)
        if n_high == 0:
            print(
                "\nNote: 0 suppliers currently flag HIGH. Check the counts above: if either "
                "condition alone is empty, the threshold (not a quiet period) is the reason. "
                "Thresholds are a business decision - change them in kpi_supplier_risk, not here."
            )

if __name__ == "__main__":
    main()
