#!/usr/bin/env bash
# Rebuild the warehouse layer in the right order. Needs DATABASE_URL-style PG* env or a URL:
#   PGHOST=localhost PGUSER=nexus_user PGPASSWORD=... PGDATABASE=nexus_db ./apply_all.sh
set -euo pipefail
cd "$(dirname "$0")"
psql -v ON_ERROR_STOP=1 -q -f schema_star.sql        # dims + facts + read-only role
psql -v ON_ERROR_STOP=1 -q -f kpi_views.sql          # KPI views
psql -v ON_ERROR_STOP=1 -q -f pii_masking.sql        # strip PII access from the agent role
echo "warehouse layer applied (schema_star, kpi_views, pii_masking)"
