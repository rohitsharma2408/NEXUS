"""
Loads the CSVs from the S3 data lake bucket into the RDS Postgres instance provisioned
by Terraform (i.e. the cloud equivalent of project2_analytics/ingestion/load_csv_to_postgres.py,
but reading from s3://<bucket>/raw/ instead of the local filesystem).

Usage:
    python load_s3_to_rds.py --bucket nexus-datalake-my-suffix
Requires DATABASE_URL to point at the RDS endpoint (see terraform output rds_endpoint)
and standard AWS credentials in the environment.
"""
import argparse
import io
import os

import boto3
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

TABLES = [
    "transactions", "customers", "returns", "products",
    "price_history", "inventory", "supplier_costs", "marketing_spend",
]

DATE_COLUMNS = {
    "transactions": ["date"],
    "customers": ["registration_date"],
    "returns": ["return_date"],
    "products": ["launch_date"],
    "inventory": ["last_restock_date"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="raw/")
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    engine = create_engine(os.environ["DATABASE_URL"])

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))

    for table in TABLES:
        key = f"{args.prefix}{table}.csv"
        obj = s3.get_object(Bucket=args.bucket, Key=key)
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))

        for col in DATE_COLUMNS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        df.to_sql(f"raw_{table}", engine, if_exists="replace", index=False, method="multi", chunksize=5000)
        print(f"Loaded s3://{args.bucket}/{key} -> raw_{table} ({len(df)} rows)")

    print("Now run schema_star.sql and kpi_views.sql against the RDS instance to finish setup:")
    print("  psql \"$DATABASE_URL\" -f project2_analytics/sql/schema_star.sql")
    print("  psql \"$DATABASE_URL\" -f project2_analytics/sql/kpi_views.sql")
    print("  psql \"$DATABASE_URL\" -f project1_agentic/rag/pgvector_setup.sql")


if __name__ == "__main__":
    main()
