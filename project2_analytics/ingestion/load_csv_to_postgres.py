"""
Stage 1 — Data Ingestion
Loads the 8 CSVs from data/raw/ into raw_* PostgreSQL tables.

Usage:
    python project2_analytics/ingestion/load_csv_to_postgres.py
    python project2_analytics/ingestion/load_csv_to_postgres.py --data-dir /app/data/raw
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

TABLES = [
    "transactions",
    "customers",
    "returns",
    "products",
    "price_history",
    "inventory",
    "supplier_costs",
    "marketing_spend",
]

DATE_COLUMNS = {
    "transactions": ["date"],
    "customers": ["registration_date"],
    "returns": ["return_date"],
    "products": ["launch_date"],
    "inventory": ["last_restock_date"],
}


def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set (check your .env)")
    return create_engine(db_url)


def load_all(data_dir: Path, engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))

    for table in TABLES:
        csv_path = data_dir / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing expected file: {csv_path}")

        df = pd.read_csv(csv_path)

        for col in DATE_COLUMNS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df.to_sql(f"raw_{table}", engine, if_exists="replace", index=False, method="multi", chunksize=5000)
        print(f"Loaded {table}: {len(df)} rows, {len(df.columns)} columns -> raw_{table}")


def basic_integrity_checks(engine) -> None:
    """Lightweight checks mirroring the blueprint's verification approach."""
    checks = {
        "returns_resolve_to_transactions": """
            SELECT COUNT(*) FROM raw_returns r
            LEFT JOIN raw_transactions t ON r.transaction_id = t.transaction_id
            WHERE t.transaction_id IS NULL
        """,
        "returns_resolve_to_customers": """
            SELECT COUNT(*) FROM raw_returns r
            LEFT JOIN raw_customers c ON r.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """,
        "returns_resolve_to_products": """
            SELECT COUNT(*) FROM raw_returns r
            LEFT JOIN raw_products p ON r.product_id = p.product_id
            WHERE p.product_id IS NULL
        """,
    }
    with engine.connect() as conn:
        for name, sql in checks.items():
            bad = conn.execute(text(sql)).scalar()
            status = "OK" if bad == 0 else f"WARNING: {bad} orphan rows"
            print(f"  integrity[{name}]: {status}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "data" / "raw"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    engine = get_engine()

    print(f"Ingesting CSVs from {data_dir} ...")
    load_all(data_dir, engine)

    print("Running referential-integrity checks ...")
    basic_integrity_checks(engine)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
