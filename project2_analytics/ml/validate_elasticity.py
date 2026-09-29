"""
The ONLY sanctioned use of price_history.price_elasticity: as a validation label,
never as a training feature (see Section 4.1 of the blueprint).

Derives elasticity from % change in units_sold vs. % change in listed_price_usd,
per product across months where price actually changed, and compares it against the
static provided value to catch a systematic (not just noisy) gap.
"""
from features import read_sql
import pandas as pd
import numpy as np


def derive_elasticity() -> pd.DataFrame:
    df = read_sql("""
        SELECT product_id, year_month, listed_price_usd, units_sold, price_elasticity
        FROM fact_price_history
        ORDER BY product_id, year_month
    """)
    df["pct_price_change"] = df.groupby("product_id")["listed_price_usd"].pct_change()
    df["pct_units_change"] = df.groupby("product_id")["units_sold"].pct_change()

    moved = df[(df["pct_price_change"].abs() > 0.001)].copy()
    moved["derived_elasticity"] = moved["pct_units_change"] / moved["pct_price_change"]
    moved = moved.replace([np.inf, -np.inf], np.nan).dropna(subset=["derived_elasticity"])

    summary = (
        moved.groupby("product_id")
        .agg(
            derived_elasticity_median=("derived_elasticity", "median"),
            n_price_moves=("derived_elasticity", "count"),
            static_price_elasticity=("price_elasticity", "first"),
        )
        .reset_index()
    )
    summary["gap"] = summary["derived_elasticity_median"] - summary["static_price_elasticity"]
    return summary


def main():
    summary = derive_elasticity()
    # A systematic gap is one that's large AND consistent, not just a couple of noisy products.
    large_gap = summary[summary["gap"].abs() > 1.0]
    print(f"Products checked: {len(summary)}")
    print(f"Products with |gap| > 1.0: {len(large_gap)}")
    if len(large_gap) > 0:
        print(large_gap.sort_values("gap", key=abs, ascending=False).head(10).to_string(index=False))
    else:
        print("No systematic gap detected — static price_elasticity looks consistent "
              "with price-driven behavior for this dataset.")


if __name__ == "__main__":
    main()
