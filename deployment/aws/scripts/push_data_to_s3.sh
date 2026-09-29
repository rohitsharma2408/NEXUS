#!/usr/bin/env bash
# Uploads data/raw/*.csv to the S3 data lake bucket created by Terraform.
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
source .env 2>/dev/null || true

: "${S3_DATA_BUCKET:?Set S3_DATA_BUCKET in .env or export it first}"
: "${AWS_REGION:=us-east-1}"

echo "Uploading data/raw/*.csv to s3://${S3_DATA_BUCKET}/raw/ ..."
aws s3 sync data/raw/ "s3://${S3_DATA_BUCKET}/raw/" --region "${AWS_REGION}"
echo "Done."
