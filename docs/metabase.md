# Metabase (BI layer)

Metabase gives analysts ad-hoc charts and dashboards on the warehouse without writing code, next
to the Streamlit KPI dashboard and the agent's inline charts.

## Start
```bash
docker compose exec -T postgres psql -U nexus_user -d postgres < project2_analytics/sql/metabase_init.sql
docker compose up -d metabase          # http://localhost:3000
```

## Connect the warehouse — as the read-only role
Admin > Databases > Add database > PostgreSQL

| Field | Value |
|---|---|
| Host / port | `postgres` / `5432` (inside compose) |
| Database | `nexus_db` |
| User | `nexus_readonly` |
| Password | the one set with `ALTER ROLE nexus_readonly PASSWORD ...` |

Connecting as `nexus_readonly` (not `nexus_user`) matters: the role was stripped of access to
customer names, e-mail addresses and raw tables by `pii_masking.sql`, so BI users get the same
PII-free surface as the agent. Metabase can query `dim_customer_masked`, the `fact_*` tables and
all `kpi_*` views. Prefer building questions on the `kpi_*` views.

## Suggested first dashboards
- Revenue and AOV by month: `kpi_revenue_by_month`
- Return rate by category: `kpi_returns_rate_by_category`
- Supplier risk: `kpi_supplier_risk`
- Marketing ROI by channel: `kpi_marketing_roi`
