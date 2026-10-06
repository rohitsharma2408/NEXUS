"""
Anomaly-detection core: seasonality-aware detectors + a labelled evaluation protocol.

Problem with the original detector
----------------------------------
IsolationForest(contamination=0.03) on raw [revenue, orders, returns, dow] always flags 3% of
days by construction, and because it sees raw revenue it mostly flags the December peak, which
is normal seasonality, not an anomaly. There was also no way to say how accurate it was: the
warehouse has no labelled anomalies.

Approach
--------
1. Remove the expected pattern: log value minus a centred 29-day rolling median (level and
   annual seasonality), minus the day-of-week effect. What is left is noise plus true anomalies.
2. Score each day with a robust z-score (median/MAD), per channel: log revenue, log orders and
   sqrt(returns). Day score = max |z| across channels. Flag |z| > Z_THRESHOLD.
3. Evaluate by injecting known anomalies of known size into the real series and measuring
   precision / recall / F1 (anomaly_eval.py). Real labels do not exist, so this measures
   detection power for stated magnitudes, not performance on unknown real-world incidents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z_THRESHOLD = 3.5
CHANNELS = ("revenue", "orders", "return_count")


def _transform(s: pd.Series, name: str) -> pd.Series:
    if name == "return_count":
        return np.sqrt(s.clip(lower=0))
    return np.log(s.clip(lower=1))


def residuals(df: pd.DataFrame) -> pd.DataFrame:
    """df indexed by date with columns revenue, orders, return_count."""
    out = {}
    for c in CHANNELS:
        y = _transform(df[c].astype(float), c)
        base = y.rolling(29, center=True, min_periods=15).median()
        r = y - base
        dow = r.groupby(r.index.dayofweek).transform("median")
        out[c] = r - dow
    return pd.DataFrame(out, index=df.index)


def robust_z(res: pd.DataFrame) -> pd.DataFrame:
    z = {}
    for c in res.columns:
        med = res[c].median()
        mad = (res[c] - med).abs().median() * 1.4826
        z[c] = (res[c] - med) / max(mad, 1e-9)
    return pd.DataFrame(z, index=res.index)


def detect_robust_z(df: pd.DataFrame, threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    z = robust_z(residuals(df))
    # A day with unusually FEW returns is not an actionable anomaly; only returns spikes are.
    z["return_count"] = z["return_count"].clip(lower=0)
    out = df.copy()
    for c in CHANNELS:
        out[f"z_{c}"] = z[c]
    out["anomaly_score"] = z.abs().max(axis=1)
    worst = z.abs().idxmax(axis=1)  # return_count is already clipped at 0, so only spikes drive it
    out["driver"] = worst
    out["direction"] = [("spike" if z.loc[i, w] > 0 else "dip") for i, w in zip(z.index, worst)]
    out["is_anomaly"] = out["anomaly_score"] > threshold
    return out


def detect_iforest_residual(df: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.Series:
    from sklearn.ensemble import IsolationForest
    res = residuals(df).dropna()
    m = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(res)
    flags = pd.Series(m.predict(res) == -1, index=res.index)
    return flags.reindex(df.index, fill_value=False)


def detect_legacy_iforest(df: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.Series:
    """Exactly what train_anomaly.py did before: raw values + day of week."""
    from sklearn.ensemble import IsolationForest
    X = pd.DataFrame({"revenue": df["revenue"], "orders": df["orders"],
                      "return_count": df["return_count"], "dow": df.index.dayofweek})
    m = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    return pd.Series(m.predict(X) == -1, index=df.index)


# ---------------------------------------------------------------- evaluation protocol
INJECTION_TYPES = {
    "volume_spike_1.6x": ("volume", 1.6),
    "volume_spike_2.0x": ("volume", 2.0),
    "volume_dip_0.5x": ("volume", 0.5),
    "returns_spike_3x": ("returns", 3.0),
    "returns_spike_4x": ("returns", 4.0),
}


def inject(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.03):
    """Return (modified df, {date: type}) with well-separated injected anomalies."""
    n = len(df)
    k = max(len(INJECTION_TYPES), int(round(frac * n)))
    k -= k % len(INJECTION_TYPES)
    chosen: list[int] = []
    for i in rng.permutation(n):
        if all(abs(int(i) - c) >= 3 for c in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    types = list(INJECTION_TYPES)
    out = df.copy()
    truth = {}
    for j, i in enumerate(chosen):
        t = types[j % len(types)]
        kind, m = INJECTION_TYPES[t]
        d = out.index[i]
        if kind == "volume":
            out.loc[d, "revenue"] *= m
            out.loc[d, "orders"] = max(1, round(out.loc[d, "orders"] * m))
        else:
            out.loc[d, "return_count"] = round(out.loc[d, "return_count"] * m + 1)
        truth[d] = t
    return out, truth


def score_flags(flags: pd.Series, truth: dict) -> dict:
    flagged = set(flags[flags].index)
    actual = set(truth)
    tp = len(flagged & actual)
    fp = len(flagged - actual)
    fn = len(actual - flagged)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    by_type = {}
    for t in INJECTION_TYPES:
        days = [d for d, tt in truth.items() if tt == t]
        by_type[t] = sum(d in flagged for d in days) / len(days) if days else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "flagged": len(flagged), "recall_by_type": by_type}
