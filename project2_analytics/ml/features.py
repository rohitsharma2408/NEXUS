"""
Shared feature-engineering helpers for Stage 4 (ML Layer).

Enforces the blueprint's Section 4.1 fairness fix in one place so every model
inherits it automatically instead of re-implementing it:
  - gender / age / country are dropped before any feature frame reaches a model.
  - small subgroups (gender=Other, country=Sweden) get a low_confidence flag.
  - price_elasticity is never allowed into a feature frame (see validate_elasticity.py
    for the only sanctioned use of that column).
"""
from __future__ import annotations

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

PROTECTED_COLUMNS = ["gender", "age", "country"]
FORBIDDEN_FEATURE_COLUMNS = ["price_elasticity"]

SMALL_SUBGROUPS = {
    "gender": {"Other"},
    "country": {"Sweden"},
}


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url)


def read_sql(query: str) -> pd.DataFrame:
    return pd.read_sql(query, get_engine())


def flag_low_confidence(df: pd.DataFrame, raw_customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a low_confidence boolean based on membership in a known small subgroup,
    computed from the *raw* customer table (which still has gender/country) and
    joined back in by customer_id, then dropped again before modeling.
    """
    flags = pd.Series(False, index=raw_customer_df.index)
    for col, values in SMALL_SUBGROUPS.items():
        if col in raw_customer_df.columns:
            flags = flags | raw_customer_df[col].isin(values)
    flagged = raw_customer_df[["customer_id"]].copy()
    flagged["low_confidence"] = flags.values
    return df.merge(flagged, on="customer_id", how="left")


def build_feature_frame(df: pd.DataFrame, strip_protected: bool = True) -> pd.DataFrame:
    """
    Call this as the LAST step before handing a dataframe to model.fit(...).
    Drops protected attributes and any accidentally-included validation-only columns.
    """
    out = df.copy()
    if strip_protected:
        out = out.drop(columns=[c for c in PROTECTED_COLUMNS if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in FORBIDDEN_FEATURE_COLUMNS if c in out.columns], errors="ignore")
    return out


def assert_no_forbidden_columns(df: pd.DataFrame) -> None:
    leaked = [c for c in PROTECTED_COLUMNS + FORBIDDEN_FEATURE_COLUMNS if c in df.columns]
    if leaked:
        raise ValueError(
            f"Refusing to train: forbidden columns present in feature frame: {leaked}. "
            "Run build_feature_frame() first."
        )
