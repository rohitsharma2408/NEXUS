"""
The ONLY sanctioned use of price_history.price_elasticity: as a validation label,
never as a training feature (see Section 4.1 of the blueprint).

Re-derives each product's price elasticity from the data and compares it with the static
provided value, to catch a systematic (not just noisy) gap.

Why the estimator changed
-------------------------
The first version took median(%change units / %change price) per product. A ratio of two
noisy month-over-month changes explodes whenever the price barely moved, so most products
showed |gap| > 1 with derived values near +/-10 from only 6-18 observations. That measured
the estimator's noise, not the data.

Now: per product, a log-log first-difference regression through the origin,
    d log(units) = beta * d log(price) + error
 * only months with a real price move (|d log price| >= MIN_MOVE) identify beta,
 * the cross-product mean of d log(units) in each month is removed first, so seasonality and
   other shocks shared by every product are not mistaken for price response,
 * every beta comes with a standard error; a gap only counts when it exceeds Z_CRIT standard
   errors, and products whose beta is too uncertain are reported as "unreliable", not "wrong".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_MOVE = 0.02      # ignore price changes smaller than ~2%
MIN_MOVES = 6        # need at least this many real price moves per product
MAX_SE = 1.0         # beta estimates less certain than this are "unreliable"
Z_CRIT = 2.0         # a gap is significant only beyond this many standard errors


def estimate_elasticity(df: pd.DataFrame) -> pd.DataFrame:
    """df columns: product_id, year_month, listed_price_usd, units_sold, price_elasticity."""
    d = df.sort_values(["product_id", "year_month"]).copy()
    d = d[(d["listed_price_usd"] > 0) & (d["units_sold"] > 0)]
    d["dlp"] = d.groupby("product_id")["listed_price_usd"].transform(lambda s: np.log(s).diff())
    d["dlu"] = d.groupby("product_id")["units_sold"].transform(lambda s: np.log(s).diff())
    d = d.dropna(subset=["dlp", "dlu"])
    # remove what every product did together that month (seasonality, market-wide shocks)
    d["dlu"] = d["dlu"] - d.groupby("year_month")["dlu"].transform("mean")

    rows = []
    for pid, g in d[d["dlp"].abs() >= MIN_MOVE].groupby("product_id"):
        n = len(g)
        x, y = g["dlp"].to_numpy(), g["dlu"].to_numpy()
        sxx = float((x * x).sum())
        if n < MIN_MOVES or sxx == 0:
            continue
        beta = float((x * y).sum() / sxx)
        resid = y - beta * x
        se = float(np.sqrt((resid ** 2).sum() / (n - 1) / sxx))
        rows.append((pid, beta, se, n, g["price_elasticity"].iloc[0]))
    out = pd.DataFrame(rows, columns=["product_id", "derived_elasticity", "se", "n_price_moves",
                                      "static_price_elasticity"])
    out["gap"] = out["derived_elasticity"] - out["static_price_elasticity"]
    out["reliable"] = out["se"] <= MAX_SE
    out["significant_gap"] = out["reliable"] & (out["gap"].abs() > Z_CRIT * out["se"])
    return out


def summarize(res: pd.DataFrame, n_products_total: int) -> dict:
    rel = res[res["reliable"]]
    s = {
        "products_total": n_products_total,
        "products_estimable": len(res),
        "products_reliable": len(rel),
        "significant_gaps": int(rel["significant_gap"].sum()),
        "share_significant": float(rel["significant_gap"].mean()) if len(rel) else float("nan"),
        "median_gap": float(rel["gap"].median()) if len(rel) else float("nan"),
        "sign_agreement": float((np.sign(rel["derived_elasticity"]) ==
                                 np.sign(rel["static_price_elasticity"])).mean()) if len(rel) else float("nan"),
        "rank_corr": float(rel["derived_elasticity"].corr(rel["static_price_elasticity"],
                                                          method="spearman")) if len(rel) > 2 else float("nan"),
    }
    return s


def verdict(s: dict) -> str:
    """Rank correlation is the main signal: per-product betas are noisy, but if the static
    column carried real information the two would still be ordered alike."""
    if s["products_reliable"] < 20:
        return ("inconclusive - not enough price variation to validate the static elasticity either "
                "way. The static column stays validation-only and unused.")
    if s["rank_corr"] >= 0.5 and s["share_significant"] <= 0.2 and abs(s["median_gap"]) <= 0.5:
        return ("consistent - static price_elasticity matches observed price response "
                "(disagreements are about what chance gives).")
    return ("systematic disagreement - the static price_elasticity does not match observed price "
            "response (weak rank agreement and/or many significant gaps). Do not use it for "
            "pricing decisions.")


def derive_elasticity() -> pd.DataFrame:
    from features import read_sql
    df = read_sql("""
        SELECT product_id, year_month, listed_price_usd, units_sold, price_elasticity
        FROM fact_price_history
        ORDER BY product_id, year_month
    """)
    return estimate_elasticity(df), df["product_id"].nunique()


def main():
    res, n_total = derive_elasticity()
    s = summarize(res, n_total)
    print(f"Products in warehouse: {s['products_total']}")
    print(f"Estimable (>= {MIN_MOVES} price moves of >= {MIN_MOVE:.0%}): {s['products_estimable']}")
    print(f"Reliable (std error <= {MAX_SE}): {s['products_reliable']}")
    print(f"Reliable products whose derived value differs from the static one by > {Z_CRIT:.0f} SE: "
          f"{s['significant_gaps']} ({s['share_significant']:.0%})")
    print(f"Median gap (derived - static): {s['median_gap']:+.2f} | "
          f"sign agreement: {s['sign_agreement']:.0%} | rank correlation: {s['rank_corr']:.2f}")
    v = verdict(s)
    print("\nVerdict:", v)
    if v.startswith("systematic"):
        big = res[res["significant_gap"]].sort_values("gap", key=abs, ascending=False).head(10)
        print(big.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
