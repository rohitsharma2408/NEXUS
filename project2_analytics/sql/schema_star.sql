-- Stage 2 — Star Schema Modeling
-- Run after load_csv_to_postgres.py has populated raw_* tables.

DROP TABLE IF EXISTS dim_customer CASCADE;
DROP TABLE IF EXISTS dim_product CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS fact_sales CASCADE;
DROP TABLE IF EXISTS fact_returns CASCADE;
DROP TABLE IF EXISTS fact_inventory CASCADE;
DROP TABLE IF EXISTS fact_supplier_costs CASCADE;
DROP TABLE IF EXISTS fact_marketing_spend CASCADE;
DROP TABLE IF EXISTS fact_price_history CASCADE;

-- ===== Dimensions =====

-- NOTE: gender/age/country are kept here (reporting/segmentation only) but MUST be
-- dropped before any table reaches the ML feature layer — see ml/features.py.
CREATE TABLE dim_customer AS
SELECT
    customer_id,
    first_name,
    last_name,
    country,
    currency,
    age,
    gender,
    registration_date,
    is_premium,
    email_verified
FROM raw_customers;
ALTER TABLE dim_customer ADD PRIMARY KEY (customer_id);

CREATE TABLE dim_product AS
SELECT
    product_id,
    name,
    category,
    brand,
    unit_price_usd,
    unit_cost_usd,
    weight_kg,
    is_active,
    launch_date
FROM raw_products;
ALTER TABLE dim_product ADD PRIMARY KEY (product_id);

CREATE TABLE dim_date AS
SELECT DISTINCT
    date::date AS date_key,
    EXTRACT(YEAR FROM date)::int AS year,
    EXTRACT(MONTH FROM date)::int AS month,
    EXTRACT(QUARTER FROM date)::int AS quarter,
    TO_CHAR(date, 'YYYY-MM') AS year_month
FROM raw_transactions;
ALTER TABLE dim_date ADD PRIMARY KEY (date_key);

-- ===== Facts =====

CREATE TABLE fact_sales AS
SELECT
    transaction_id,
    customer_id,
    product_id,
    date::date AS order_date,
    quantity,
    unit_price_usd,
    discount_pct,
    revenue_usd,
    cost_usd,
    profit_usd,
    shipping_cost_usd,
    channel,
    payment_method,
    status,
    country,
    category
FROM raw_transactions;
ALTER TABLE fact_sales ADD PRIMARY KEY (transaction_id);
CREATE INDEX idx_fact_sales_customer ON fact_sales(customer_id);
CREATE INDEX idx_fact_sales_product ON fact_sales(product_id);
CREATE INDEX idx_fact_sales_date ON fact_sales(order_date);

CREATE TABLE fact_returns AS
SELECT
    return_id,
    transaction_id,
    customer_id,
    product_id,
    return_date::date AS return_date,
    reason,
    refund_amount_usd,
    restocked
FROM raw_returns;
ALTER TABLE fact_returns ADD PRIMARY KEY (return_id);
CREATE INDEX idx_fact_returns_txn ON fact_returns(transaction_id);

CREATE TABLE fact_inventory AS
SELECT
    product_id,
    category,
    stock_units,
    reorder_point,
    warehouse_location,
    last_restock_date::date AS last_restock_date,
    supplier_lead_days
FROM raw_inventory;
CREATE INDEX idx_fact_inventory_product ON fact_inventory(product_id);

CREATE TABLE fact_supplier_costs AS
SELECT
    product_id,
    category,
    supplier_name,
    supplier_rank,
    unit_cost_usd,
    ordering_cost_usd,
    annual_holding_cost_usd,
    holding_cost_pct,
    lead_time_days,
    min_order_qty,
    reliability_score,
    is_primary
FROM raw_supplier_costs;
CREATE INDEX idx_fact_supplier_product ON fact_supplier_costs(product_id);

CREATE TABLE fact_marketing_spend AS
SELECT
    year_month,
    channel,
    spend_usd,
    impressions,
    clicks,
    ctr,
    actual_orders,
    actual_customers,
    actual_revenue_usd,
    roas,
    cac_usd,
    cost_per_order_usd
FROM raw_marketing_spend;

-- price_elasticity is kept here for validation ONLY — never feed it to a model as a
-- feature (see ml/features.py and ml/validate_elasticity.py).
CREATE TABLE fact_price_history AS
SELECT
    product_id,
    category,
    year_month,
    listed_price_usd,
    base_price_usd,
    competitor_price_usd,
    price_index,
    is_promotional,
    price_elasticity,
    units_sold,
    revenue_usd,
    margin_pct
FROM raw_price_history;
CREATE INDEX idx_fact_price_product ON fact_price_history(product_id);

-- ===== Enterprise safety: read-only role for the agentic (Project 1) layer =====
-- Matches Diagram 7 (Enterprise Safety Flow): the SQL Agent only ever connects as this role.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'nexus_readonly') THEN
        CREATE ROLE nexus_readonly LOGIN PASSWORD 'change_me_too';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE nexus_db TO nexus_readonly;
GRANT USAGE ON SCHEMA public TO nexus_readonly;
GRANT SELECT ON dim_customer, dim_product, dim_date,
                fact_sales, fact_returns, fact_inventory,
                fact_supplier_costs, fact_marketing_spend
    TO nexus_readonly;
-- fact_price_history is intentionally NOT granted to the agent role by default;
-- the elasticity column is validation-only. Uncomment if the agent needs pricing facts
-- minus elasticity (create a view instead of granting the raw table).
CREATE OR REPLACE VIEW v_price_history_for_agents AS
    SELECT product_id, category, year_month, listed_price_usd, base_price_usd,
           competitor_price_usd, price_index, is_promotional, units_sold, revenue_usd, margin_pct
    FROM fact_price_history;
GRANT SELECT ON v_price_history_for_agents TO nexus_readonly;

ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO nexus_readonly;
