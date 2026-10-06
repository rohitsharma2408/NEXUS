# NEXUS — Agentic Business Intelligence Platform

**An AI analyst that investigates why, not just what — built on a real data warehouse, with document-grounded evidence and an honest confidence level on every answer.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal) ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)

## The problem this solves

Most "chat with your data" demos stop at translating a question into SQL. NEXUS goes a step further: a supervisor routes a business question to a SQL agent (warehouse), an ML agent (trained models) and a RAG agent (business documents in pgvector). An investigation step merges the results, an evidence checker sets a confidence level and flags missing corroboration, and a business analyst writes the answer, separating correlation from causation.

## Architecture

```mermaid
flowchart TD
    Q["User question"] --> S["Supervisor<br/>(intent routing)"]

    S --> SQL["SQL Agent"]
    S --> ML["ML Agent"]
    S --> RAG["RAG Agent"]

    SQL --> DB[("PostgreSQL<br/>warehouse")]
    ML --> MODELS[("Forecast, Churn,<br/>Anomaly, Supplier-risk")]
    RAG --> VEC[("pgvector<br/>policies, reports")]

    DB --> INV["Investigation Agent"]
    MODELS --> INV
    VEC --> INV

    INV --> EC["Evidence Checker<br/>(confidence level + caveats)"]
    EC --> BA["Business Analyst"]
    BA --> OUT["Answer + Evidence + Confidence"]

    classDef store fill:#1a1a2e,stroke:#666,color:#eee
    class DB,MODELS,VEC store
```

*Not built on purpose: MCP and the AWS deployment (blueprint places them last). The diagram above
reflects what actually runs.*

## Status (only what has been run and observed)

| Component | Status | Notes |
|---|---|---|
| Ingestion + star schema (`dim_*`, `fact_*`) | Working | 8 CSVs, 100,000 transactions, referential checks pass |
| KPI views + Streamlit dashboard | Working | Sales, customers, returns, operations, marketing |
| `/investigate` (SQL + RAG, Gemini) | Working | Answers grounded in warehouse rows and retrieved documents |
| Ask tab in the dashboard | Working | Calls `/investigate` from inside the dashboard |
| ML models | Trained, baseline quality | See metrics below; not yet tuned or wired into routing |
| Supervisor | Plain Python by default; LangGraph opt-in | `USE_LANGGRAPH=1` fans SQL / ML / RAG / drill-down out in parallel. Same step functions in both modes; tests run both |
| Evidence Checker | Working | Verifies the actual numbers in the answer against retrieved evidence (`number_verifier.py`), not just source counts |
| PII masking | Working, three layers | DB role has no access to names / e-mail / raw tables (`pii_masking.sql`); validator blocks PII columns and tables; output scrubbed. Tested against the live role |
| Audit logging | Working | Append-only, hash-chained `agent_audit_log`; every SQL attempt, ML call, retrieval and answer; `/audit/verify` and `/audit/recent` (admin token) |
| Multi-step investigation | Working | "Why did X change" runs a deterministic decomposition by country / category / channel plus a same-month-in-other-years check (`investigation_playbook.py`) |
| Chart output from the agent | Working | Chart specs returned by `/investigate` and rendered in the dashboard Ask tab |
| SQL agents | One shared core | `nl2sql_core.py` is used by both `sql_agent.py` (Postgres) and the BI-Bench runner (SQLite) |
| Metabase | Compose service + docs | `docker compose up -d metabase`; connect as `nexus_readonly`. See `docs/metabase.md` |
| Internal benchmark | 26 SQL questions with executable ground truth + safety set | `python evaluation/internal_benchmark.py validate` (no LLM needed). LLM accuracy needs an API key: `... agent` |
| BI-Bench | 30% (6/20), one pilot run | 20-case size-capped pilot, baseline agent, no data-management tools. See `docs/bibench_pilot.md`. Comparable paper baselines: GPT-4o 27.1%, DeepSeek-V4-Pro 23.5% (SQL, no-tool) |
| AWS deployment | Not deployed | Terraform written, never applied |

### Measured ML baselines (single run, time-based split for forecasting)

| Model | Result |
|---|---|
| Demand forecast (seasonal level x index + XGBoost blend) | Walk-forward MAE 5.78 vs 8.09 for last-month-carried-forward (+28.6% skill); the previous model scored 8.81, worse than naive. See `docs/forecast_backtest.md` |
| Churn (Random Forest) | ROC-AUC 0.730, F1 0.661 (time-based split) |
| Anomaly detection (robust z-score on seasonality-removed residuals) | F1 0.79 vs 0.39 for the old Isolation Forest on planted anomalies; 6 of 1,096 real days flagged. Real data has no labels, so accuracy is measured by planting known anomalies in the real series. See `docs/anomaly_eval.md` |
| Supplier risk | Rule-based (SQL view), not a classifier | Previous classifier was circular (label and features overlapped); see `train_supplier_risk.py` |

These are first-pass numbers, not production claims. Product-month sales are noisy counts, so single-product forecasts stay noisy; category totals are far more reliable.

## Key design decisions

- **`price_elasticity` is never a model feature.** It is a fixed constant per product in the raw data, so training on it would leak the label. It is used only as a validation check.
- **The static `price_elasticity` does not match observed price response.** `project2_analytics/ml/validate_elasticity.py` found 498 products with enough price changes to estimate; among the 107 with reliable estimates, 46% differ significantly from the static value, sign agreement is 52%, and rank correlation is -0.04. Treat it as an assumed placeholder, not a measured elasticity, and do not use it for pricing decisions. Many derived values are positive, which points to confounding (promotions, seasonality), so the derived numbers are not a drop-in replacement either.
- **Protected attributes never reach a model.** `gender`, `age`, `country` are stripped before feature frames are built; small subgroups get a low-confidence flag.
- **The SQL agent is read-only by construction:** a dedicated Postgres role with SELECT-only grants, plus a validator that blocks write keywords, unknown tables and queries without a LIMIT, with a statement timeout.
- **Confidence is evidence-based.** It rises when SQL and document evidence agree, and the analyst is told to separate correlation from causation. It does not perform formal causal inference.

## Data (not included in this repo)

The dataset is not committed. Download **Global E-Commerce & Supply Chain Database** from Kaggle (`parsakh/global-e-commerce-and-supply-chain-database`) and place the 8 CSVs in `data/raw/`:

```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json
kaggle datasets download -d parsakh/global-e-commerce-and-supply-chain-database
unzip global-e-commerce-and-supply-chain-database.zip -d data/raw/
```

## Quick start

```bash
git clone https://github.com/rohitsharma2408/NEXUS.git && cd NEXUS
cp .env.example .env        # set LLM_PROVIDER, LLM_MODEL and the matching API key
docker compose up -d --build

# load data and build the warehouse (psql lives in the postgres container; -T is required for piped input)
docker compose exec api python project2_analytics/ingestion/load_csv_to_postgres.py
docker compose exec -T postgres psql -U nexus_user -d nexus_db < project2_analytics/sql/schema_star.sql
docker compose exec -T postgres psql -U nexus_user -d nexus_db < project2_analytics/sql/kpi_views.sql
docker compose exec -T postgres psql -U nexus_user -d nexus_db < project2_analytics/sql/pii_masking.sql   # strips PII access from the agent role
docker compose exec -T postgres psql -U nexus_user -d nexus_db < project1_agentic/rag/pgvector_setup.sql

# train models, then load the sample business documents for RAG
docker compose exec api python project2_analytics/ml/train_all.py
docker compose exec api python project1_agentic/rag/ingest_documents.py
```

- Dashboard: http://localhost:8501 (start on the **Ask the Analyst** tab)
- API docs: http://localhost:8000/docs

**LLM providers:** set `LLM_PROVIDER` to `gemini`, `groq`, `openai` or `anthropic`. Model names change; list what your key can use rather than guessing. The Gemini free tier is rate-limited (each question makes several LLM calls).

**Data range:** the dataset spans Jan 2022 – Dec 2024, so ask about specific periods, not "last month".

## Project structure

```
project2_analytics/   ingestion, star schema + KPI SQL, ML training, dashboard
project1_agentic/     supervisor, SQL/ML/RAG agents, evidence checker, FastAPI app, RAG documents
evaluation/           internal benchmark + BI-Bench pilot runner (20-case pilot: 30%)
deployment/           Dockerfiles, AWS Terraform, CI/CD
docs/                 architecture notes
```

## Roadmap

- [ ] Real BI dashboards (Metabase) alongside the fixed Streamlit views
- [x] SQL agent retries failed queries with the real error fed back (main app + BI-Bench pilot)
- [ ] Agent returns charts; multi-step investigations
- [ ] Rework the ML layer (proper labels, tuning, saved metrics, persistent MLflow)
- [ ] Internal benchmark with ground-truth answers
- [x] BI-Bench pilot (20 cases, 30%) — scale to the ~50 cases under 25MB, or run "+tools" mode
- [ ] Deploy to AWS with the existing Terraform (roughly $30–60/month if left running)
