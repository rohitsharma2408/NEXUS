"""
Demand-forecast core: ONE feature builder shared by training, backtesting and inference.

Why this exists
---------------
The previous forecaster had three problems that explain why it "barely beat guessing":
  1. No seasonality. `month_num` was a 0..35 row counter, not a calendar feature, even though
     this warehouse has a strong annual cycle (December ~1.6x an average month, February ~0.6x).
  2. Train/serve skew. At inference `ml_agent` hard-coded month_num=1 and used units(t) as
     "lag_units_sold", while training used units(t-1). The model was fed inputs it never saw.
  3. No baselines, so there was no way to tell whether the model added anything.

Design
------
For every product and every origin month t we predict units sold in month t+1 using only
information available at t:
  * recent level: lag1..lag3, rolling means over 3 / 6 / 12 months
  * same month last year: units(t+1-12)
  * calendar: target month as sin/cos + the *seasonal index* of that month (global and per
    category), estimated from the training window only (no peeking at the test period)
  * promo behaviour: share of promotional months in the trailing 12, price_index at t
No target-month price or promo flag is used, because those are not known when forecasting.

Candidates evaluated side by side in forecast_backtest.py:
  naive, moving_avg_3, moving_avg_12, seasonal_naive, level_x_season, legacy_xgb, xgb_seasonal
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HISTORY_MIN = 3  # months of history needed before a row is usable

FEATURES = [
    "lag1", "lag2", "lag3", "roll3", "roll6", "roll12", "lag12_target",
    "month_sin", "month_cos", "target_month",
    "season_global", "season_category", "level_x_season",
    "promo_rate_12", "price_index_t", "cat_code",
]

LEGACY_FEATURES = [
    "listed_price_usd", "base_price_usd", "competitor_price_usd",
    "price_index", "is_promotional", "month_num", "lag_units_sold",
]


def ym_to_idx(ym: pd.Series) -> pd.Series:
    """'2024-03' -> 2024*12 + 2 (a monotone integer month index)."""
    y = ym.str.slice(0, 4).astype(int)
    m = ym.str.slice(5, 7).astype(int)
    return y * 12 + (m - 1)


def idx_to_ym(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


@dataclass
class SeasonalIndex:
    """Multiplicative calendar-month index (1.0 = average month), global and per category."""
    global_idx: dict = field(default_factory=dict)       # month(1-12) -> float
    category_idx: dict = field(default_factory=dict)     # (category, month) -> float

    @classmethod
    def fit(cls, panel: pd.DataFrame, upto_idx: int | None = None) -> "SeasonalIndex":
        d = panel if upto_idx is None else panel[panel["midx"] <= upto_idx]
        d = d.assign(cal_month=d["midx"] % 12 + 1)
        g = d.groupby("cal_month")["units_sold"].mean()
        g = g / g.mean()
        out = cls(global_idx=g.to_dict())
        # per-category, shrunk toward the global index so thin categories don't get noisy
        for cat, dc in d.groupby("category"):
            c = dc.groupby("cal_month")["units_sold"].mean()
            if len(c) < 12:
                continue
            c = c / c.mean()
            n = dc.groupby("cal_month").size()
            w = n / (n + 50.0)
            for m in range(1, 13):
                out.category_idx[(cat, m)] = float(w[m] * c[m] + (1 - w[m]) * g.get(m, 1.0))
        return out

    def get_global(self, month: int) -> float:
        return float(self.global_idx.get(month, 1.0))

    def get_category(self, category: str, month: int) -> float:
        return float(self.category_idx.get((category, month), self.get_global(month)))


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """raw = fact_price_history rows. Adds the integer month index and tidy dtypes."""
    p = raw.copy()
    p["midx"] = ym_to_idx(p["year_month"])
    p["is_promotional"] = p["is_promotional"].astype(int)
    p["units_sold"] = p["units_sold"].astype(float)
    return p.sort_values(["product_id", "midx"]).reset_index(drop=True)


def make_features(panel: pd.DataFrame, season: SeasonalIndex, categories: list[str] | None = None,
                  with_target: bool = True) -> pd.DataFrame:
    """One row per (product, origin month t). Features use data up to and including t only."""
    cats = categories or sorted(panel["category"].unique())
    cat_map = {c: i for i, c in enumerate(cats)}
    g = panel.groupby("product_id", sort=False)
    out = panel[["product_id", "category", "year_month", "midx"]].copy()
    u = panel["units_sold"]
    out["lag1"] = u
    out["lag2"] = g["units_sold"].shift(1)
    out["lag3"] = g["units_sold"].shift(2)
    out["roll3"] = g["units_sold"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    out["roll6"] = g["units_sold"].transform(lambda s: s.rolling(6, min_periods=3).mean())
    out["roll12"] = g["units_sold"].transform(lambda s: s.rolling(12, min_periods=6).mean())
    # units in the month we are about to forecast, one year earlier = units(t+1-12) = shift(11)
    out["lag12_target"] = g["units_sold"].shift(11)
    tm = (out["midx"] + 1) % 12 + 1                      # calendar month of the target
    out["target_month"] = tm
    out["month_sin"] = np.sin(2 * np.pi * tm / 12)
    out["month_cos"] = np.cos(2 * np.pi * tm / 12)
    out["season_global"] = tm.map(season.get_global)
    out["season_category"] = [season.get_category(c, m) for c, m in zip(out["category"], tm)]
    out["level_x_season"] = out["roll12"].fillna(out["roll6"]) * out["season_category"]
    out["promo_rate_12"] = g["is_promotional"].transform(lambda s: s.rolling(12, min_periods=3).mean())
    out["price_index_t"] = panel["price_index"]
    out["cat_code"] = out["category"].map(cat_map)
    if with_target:
        out["target"] = g["units_sold"].shift(-1)
    out = out[out["roll3"].notna()]
    return out.reset_index(drop=True)


def make_legacy_features(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """Exactly what the old train_forecasting.py built, kept ONLY so the backtest can show
    before/after on the same rows."""
    df = raw_panel.copy()
    df["month_num"] = df.groupby("product_id").cumcount()
    df["lag_units_sold"] = df.groupby("product_id")["units_sold"].shift(1)
    df["target"] = df.groupby("product_id")["units_sold"].shift(-1)
    return df.dropna(subset=["lag_units_sold", "target"])


# ---------------------------------------------------------------- baselines
def baseline_predictions(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every baseline reads only columns that exist at origin month t."""
    return {
        "naive_last_month": f["lag1"].to_numpy(),
        "moving_avg_3": f["roll3"].to_numpy(),
        "moving_avg_12": f["roll12"].fillna(f["roll6"]).to_numpy(),
        "seasonal_naive": f["lag12_target"].fillna(f["roll12"]).fillna(f["roll6"]).to_numpy(),
        "level_x_season": f["level_x_season"].to_numpy(),
    }


# ---------------------------------------------------------------- model
def fit_xgb(train: pd.DataFrame, seed: int = 42):
    from xgboost import XGBRegressor
    m = XGBRegressor(
        objective="count:poisson", n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
        random_state=seed, n_jobs=2,
    )
    m.fit(train[FEATURES], train["target"])
    return m


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(p) & np.isfinite(y)
    y, p = y[ok], p[ok]
    err = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / max(np.sum(np.abs(y)), 1e-9)),
        "bias": float(np.mean(err)),
    }


def aggregate_wape(f: pd.DataFrame, pred: np.ndarray, by=("category", "midx")) -> float:
    """Error after summing products into category-months — the level a planner acts on."""
    d = f[list(by)].copy()
    d["y"], d["p"] = f["target"].to_numpy(), pred
    a = d.groupby(list(by))[["y", "p"]].sum()
    return float(np.sum(np.abs(a["p"] - a["y"])) / max(np.sum(a["y"]), 1e-9))
