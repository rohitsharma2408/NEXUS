# NEXUS Deployment Guide

Two deployment targets are covered here: local Docker Compose (dev), and AWS (S3 + RDS +
ECR + ECS Fargate + ALB), which is what "put the files in a cloud database and run it on a
platform" maps to for this project.

## A. Local (Docker Compose)

```bash
cp .env.example .env      # edit values
docker compose up -d --build
docker compose exec api python project2_analytics/ingestion/load_csv_to_postgres.py
docker compose exec api psql "$DATABASE_URL" -f project2_analytics/sql/schema_star.sql
docker compose exec api psql "$DATABASE_URL" -f project2_analytics/sql/kpi_views.sql
docker compose exec api psql "$DATABASE_URL" -f project2_analytics/sql/pii_masking.sql
docker compose exec api psql "$DATABASE_URL" -f project1_agentic/rag/pgvector_setup.sql
docker compose exec api python project2_analytics/ml/train_all.py
```

## B. AWS (cloud database + hosted platform)

This provisions:
- **S3** — data lake bucket holding the raw CSVs (`data/raw/*.csv`)
- **RDS PostgreSQL** — the cloud database the warehouse and agents run against (pgvector-capable)
- **ECR** — Docker image registry for the API and dashboard
- **ECS Fargate + ALB** — the "platform" the containers actually run on, behind a public load balancer

### 1. Prerequisites
- AWS CLI configured (`aws configure`) with a role that can create VPC/RDS/ECS/S3/ECR/IAM resources
- Terraform >= 1.5
- Docker
- A Secrets Manager secret containing your Anthropic API key (referenced by
  `anthropic_api_key_secret_arn` in `terraform.tfvars`)

### 2. Provision the infrastructure

```bash
cd deployment/aws/terraform
cat > terraform.tfvars <<EOF
aws_region                    = "us-east-1"
s3_data_bucket                = "nexus-datalake-yourname-123"
db_master_password            = "REPLACE_ME_STRONG"
readonly_password             = "REPLACE_ME_STRONG_2"
anthropic_api_key_secret_arn  = "arn:aws:secretsmanager:us-east-1:123456789012:secret:nexus/anthropic-key"
EOF

terraform init
terraform apply
```

Note the outputs: `alb_dns_name`, `rds_endpoint`, `ecr_api_repository_url`,
`ecr_dashboard_repository_url`, `s3_data_bucket`.

### 3. Load the data into the cloud database

```bash
cd ../scripts
export S3_DATA_BUCKET=nexus-datalake-yourname-123
export AWS_REGION=us-east-1
./push_data_to_s3.sh                       # data/raw/*.csv -> S3

export DATABASE_URL="postgresql://nexus_user:REPLACE_ME_STRONG@<rds_endpoint>:5432/nexus_db"
python load_s3_to_rds.py --bucket "$S3_DATA_BUCKET"

psql "$DATABASE_URL" -f ../../../project2_analytics/sql/schema_star.sql
psql "$DATABASE_URL" -f ../../../project2_analytics/sql/kpi_views.sql
psql "$DATABASE_URL" -f ../../../project2_analytics/sql/pii_masking.sql
psql "$DATABASE_URL" -f ../../../project1_agentic/rag/pgvector_setup.sql
```

### 4. Build, push, and run the app on ECS

```bash
export AWS_ACCOUNT_ID=123456789012
./deploy.sh
```

This builds `deployment/Dockerfile.api` and `deployment/Dockerfile.dashboard`, pushes both
to the ECR repos Terraform created, and forces a new ECS Fargate deployment behind the ALB.

Open `http://<alb_dns_name>/` for the dashboard and `http://<alb_dns_name>/docs` for the
agent API's Swagger UI.

### 5. Continuous deployment

`deployment/github_actions/ci-cd.yml` (also copied to `.github/workflows/ci-cd.yml`) runs the
same build/push/deploy steps automatically on every push to `main`. Set these repo secrets:
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`.

### 6. Cost / cleanup

`db.t4g.micro` + one Fargate task each for API/dashboard + one ALB is intentionally small
for a portfolio/demo deployment. Tear everything down with:

```bash
cd deployment/aws/terraform
terraform destroy
```
