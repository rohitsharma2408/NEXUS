-- Stage 3b — PII masking for the agentic layer.   Run AFTER schema_star.sql and kpi_views.sql.
--
-- Why this exists: the agent role used to be able to SELECT first_name / last_name (dim_customer)
-- and e-mail addresses (raw_customers). `ALTER DEFAULT PRIVILEGES ... GRANT SELECT ON TABLES`
-- in schema_star.sql also silently granted every raw_* table and fact_price_history
-- (price_elasticity is validation-only) to nexus_readonly, overriding the explicit grant list.
--
-- Fix: remove the blanket grants, expose a masked view, and grant only an explicit allow-list.
-- The SQL validator in project1_agentic/agents/sql_agent.py enforces the same list, but THIS
-- file is the real control: even a query that slips past the validator cannot read PII.

ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM nexus_readonly;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM nexus_readonly;

-- No names, no e-mail. Demographics stay because segmentation questions need them.
CREATE OR REPLACE VIEW dim_customer_masked AS
SELECT customer_id, country, currency, age, gender, registration_date, is_premium, email_verified
FROM dim_customer;

GRANT SELECT ON
    dim_customer_masked, dim_product, dim_date,
    fact_sales, fact_returns, fact_inventory, fact_supplier_costs, fact_marketing_spend,
    v_price_history_for_agents,
    kpi_revenue_by_month, kpi_returns_rate_by_category, kpi_inventory_turnover,
    kpi_supplier_risk, kpi_marketing_roi, kpi_customer_cohort, kpi_revenue_by_country_month
TO nexus_readonly;

-- The RAG agent reads business documents through the same role (no personal data in them).
DO $$ BEGIN
    IF to_regclass('public.business_documents') IS NOT NULL THEN
        GRANT SELECT ON business_documents TO nexus_readonly;
    END IF;
END $$;
