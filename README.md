# NEXUS — AI Business Analyst

Implementation of the NEXUS Master Blueprint (v6): a two-project system that turns the
**Global E-Commerce & Supply Chain Database** into a queryable warehouse (Project 2), then
puts an agentic AI Business Analyst on top of it (Project 1), and scores that analyst against
the external **BI-Bench** benchmark.

```
nexus/
├── data/raw/                    8 source CSVs (your uploaded dataset, already included)
├── project2_analytics/          Foundation: warehouse, KPIs, ML models, dashboard
│   ├── ingestion/                CSV -> PostgreSQL loader
│   ├── sql/                      star schema + KPI views
│   ├── ml/                       forecasting / churn / anomaly / supplier-risk training
│   └── dashboard/                Streamlit dashboard
├── project1_agentic/            AI layer: SQL / ML / RAG agents + supervisor + API
│   ├── agents/                   LangGraph-style agents (supervisor, sql, ml, rag,
│   │                             investigation, evidence checker, business analyst)
│   ├── api/                      FastAPI app exposing /investigate
│   └── rag/                      pgvector setup + document ingestion
├── evaluation/                   Internal benchmark + BI-Bench runner
├── deployment/                   Docker + AWS (Terraform) + CI/CD
└── docs/                         Architecture notes
```

## 1. Local quick start (Docker Compose)

```bash
cp .env.example .env                 # fill in OPENAI/ANTHROPIC key etc.
docker compose up -d --build         # postgres + api + dashboard
docker compose exec api python project2_analytics/ingestion/load_csv_to_postgres.py
docker compose exec api psql -U nexus_user -d nexus_db -f project2_analytics/sql/schema_star.sql
docker compose exec api psql -U nexus_user -d nexus_db -f project2_analytics/sql/kpi_views.sql
docker compose exec api python project2_analytics/ml/train_all.py
```

- Dashboard: http://localhost:8501
- Agent API: http://localhost:8000/docs

## 2. What each stage does

| Stage | Component | Command |
|---|---|---|
| Ingestion | `project2_analytics/ingestion/load_csv_to_postgres.py` | loads the 8 CSVs into `raw_*` tables |
| Warehouse | `project2_analytics/sql/schema_star.sql` | builds `dim_*` / `fact_*` star schema |
| KPIs | `project2_analytics/sql/kpi_views.sql` | revenue, AOV, churn, returns, inventory, supplier, marketing views |
| ML | `project2_analytics/ml/train_all.py` | demand forecast, churn, anomaly, supplier risk models -> `models/` |
| Dashboard | `project2_analytics/dashboard/app.py` | Streamlit views over the KPI layer |
| Agents | `project1_agentic/agents/*.py` | SQL / ML / RAG / Investigation / Evidence / Analyst |
| API | `project1_agentic/api/main.py` | FastAPI `/investigate` endpoint, orchestrates the agents |
| Evaluation | `evaluation/internal_benchmark.py`, `evaluation/bibench_runner.py` | scoring |
| Deployment | `deployment/` | Docker images, AWS Terraform, GitHub Actions CI/CD |

## 3. Fairness / data-safety guardrails (from the blueprint, enforced in code)

- `project2_analytics/ml/features.py` drops `gender`, `age`, `country` before any model
  ever sees a feature table (`build_feature_frame(..., strip_protected=True)`), and flags
  small subgroups (`gender=Other`, `country=Sweden`) with a `low_confidence` marker.
- `price_elasticity` from `price_history.csv` is never included as a model feature — it is
  only used in `project2_analytics/ml/validate_elasticity.py` as a sanity check against a
  price/units-sold-derived estimate.
- The agentic SQL layer is **read-only**: see `deployment/aws/terraform/main.tf` (RDS user
  grants) and `project1_agentic/agents/sql_agent.py` (query validator rejects
  DROP/DELETE/UPDATE/INSERT/ALTER, enforces row limits + timeouts).

## 4. Cloud deployment (AWS)

See `deployment/README.md` for the full walkthrough. Short version:

```bash
cd deployment/aws/scripts
./push_data_to_s3.sh              # uploads data/raw/*.csv to an S3 data lake bucket
cd ../terraform
terraform init && terraform apply  # provisions RDS Postgres + ECR + ECS Fargate + ALB
python ../scripts/load_s3_to_rds.py  # loads the CSVs from S3 into the new RDS instance
```

CI/CD (`deployment/github_actions/ci-cd.yml`) builds the API + dashboard images, pushes to
ECR, and updates the ECS services on every push to `main`.

## 5. BI-Bench evaluation

`evaluation/bibench_runner.py` clones/consumes `github.com/Hu-Chuxuan/bi-agent`, runs NEXUS's
`/investigate` endpoint against each case's natural-language question, and scores it against
that case's ground truth. This is external validation only — see the blueprint's scope note:
a BI-Bench score proves general BI navigation skill, not that the agents understand *this*
company. Do not conflate the two when reporting results.
