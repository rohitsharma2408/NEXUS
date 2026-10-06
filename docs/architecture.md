# NEXUS — What Was Built

This maps every component in `NEXUS_Blueprint_v6.pdf` to actual code in this repo, using
the real column names from your uploaded `archive__8_.zip` dataset (100,000 transactions,
8,000 customers, 500 products — figures match the blueprint's verified stats).

| Blueprint section | Code |
|---|---|
| 4.4 Load CSVs into PostgreSQL | `project2_analytics/ingestion/load_csv_to_postgres.py` |
| 4.5 Star schema | `project2_analytics/sql/schema_star.sql` |
| 4.6 KPI SQL | `project2_analytics/sql/kpi_views.sql` |
| 4.1 Fairness fix (drop gender/age/country, flag small subgroups) | `project2_analytics/ml/features.py` |
| 4.1 Elasticity warning (validation-only, never a feature) | `project2_analytics/ml/validate_elasticity.py`, enforced in every `train_*.py` |
| 9. ML layer (forecasting, churn, anomaly, supplier risk) | `project2_analytics/ml/train_*.py` |
| Diagram 3 Stage 5 — Dashboards | `project2_analytics/dashboard/app.py` |
| Diagram 4 — SQL/ML/RAG agents + supervisor | `project1_agentic/agents/*.py` |
| Diagram 7 — Enterprise safety (read-only, validator, permission layer) | `project1_agentic/agents/sql_agent.py`, `schema_star.sql` (nexus_readonly role) |
| Section 11 Project 1 output — final API | `project1_agentic/api/main.py` (`POST /investigate`) |
| Section 19 — Internal benchmark | `evaluation/internal_benchmark.py` |
| Section 5 — BI-Bench external evaluation | `evaluation/bibench_runner.py` |
| Section 20 — Productionization (Docker/CI-CD/AWS) | `deployment/` |

## Design choices worth knowing about

- **Real column names, not the blueprint's simplified ones.** Your actual CSVs use
  `revenue_usd`, `date`, `stock_units`, etc. rather than the blueprint's `revenue`,
  `order_date`, `stock_level`. All SQL/Python in this repo was written against your real
  files (verified in `data/raw/`), not the blueprint's prose description.
- **Supervisor is framework-light.** `project1_agentic/agents/supervisor.py` implements the
  routing/orchestration logic without a hard LangGraph dependency, so it's easy to run and
  test locally. Wrapping `investigate()` in a `langgraph.StateGraph` for true parallel agent
  execution and multi-turn state is a natural next step — the agent functions are already
  pure enough to drop straight into graph nodes.
- **RAG is wired but needs your documents.** `project1_agentic/rag/ingest_documents.py`
  expects `.txt` files (policies, promo calendars, ops reports) in
  `project1_agentic/rag/documents/`. None are included since the blueprint's dataset is
  structured data only — add your own to activate the RAG Agent.
- **MLflow tracking is local by default.** Set `MLFLOW_TRACKING_URI` to point at a hosted
  MLflow server if you want run history to survive container restarts.

## Honest limitations (things a first pass leaves out)

- The Terraform stack uses the account's default VPC and public subnets for simplicity —
  fine for a demo/portfolio deployment, not a production security posture (no private
  subnets/NAT gateway, no TLS on the ALB listener).
- `evaluation/bibench_runner.py` scores against ground-truth CSVs with a simple
  numeric-token-overlap heuristic, not BI-Agent's own accuracy comparator — treat it as a
  starting point, not the paper's actual metric.
- The RAG Agent's `_embed()` only implements OpenAI embeddings; swap in Anthropic's or a
  local embedder if you'd rather not depend on OpenAI for that one piece.


## Added since the first build

| Capability | Where |
|---|---|
| PII masking (DB role + validator + output scrub) | `project2_analytics/sql/pii_masking.sql`, `agents/pii.py`, `agents/sql_agent.py` |
| Audit logging (append-only, hash-chained) | `agents/audit.py`, `/audit/*` in `api/main.py` |
| Seasonal demand forecast + walk-forward backtest | `ml/forecast_core.py`, `ml/train_forecasting.py`, `evaluation/forecast_backtest.py` |
| Anomaly detection + labelled-injection evaluation | `ml/anomaly_core.py`, `evaluation/anomaly_eval.py` |
| Internal benchmark with executable ground truth | `evaluation/internal_benchmark.py`, `evaluation/internal/questions.json` |
| One shared NL-to-SQL core | `agents/nl2sql_core.py` (used by `sql_agent.py` and `evaluation/bibench/runner.py`) |
| Multi-step investigation | `agents/investigation_playbook.py` |
| LangGraph supervisor (parallel fan-out) | `agents/supervisor.py` (`USE_LANGGRAPH=1`) |
| Charts | `agents/charts.py`, dashboard Ask tab |
| Metabase | `docker-compose.yml`, `docs/metabase.md` |
