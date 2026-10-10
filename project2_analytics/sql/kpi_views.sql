-- Stage 3 — KPI Layer
-- Run after schema_star.sql.

CREATE OR REPLACE VIEW kpi_revenue_by_month AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(revenue_usd) AS total_revenue,
    SUM(profit_usd) AS total_profit,
    COUNT(DISTINCT transaction_id) AS order_count,
    ROUND((SUM(revenue_usd) / NULLIF(COUNT(DISTINCT transaction_id), 0))::numeric, 2) AS aov
FROM fact_sales
WHERE status = 'completed' OR status IS NULL
GROUP BY 1
ORDER BY 1;

CREATE OR REPLACE VIEW kpi_returns_rate_by_category AS
SELECT
    p.category,
    COUNT(r.return_id) AS return_count,
    COUNT(DISTINCT s.transaction_id) AS total_sales,
    ROUND(
        COUNT(r.return_id)::numeric / NULLIF(COUNT(DISTINCT s.transaction_id), 0) * 100, 2
    ) AS return_rate_pct
FROM fact_sales s
LEFT JOIN fact_returns r ON r.transaction_id = s.transaction_id
JOIN dim_product p ON s.product_id = p.product_id
GROUP BY 1
ORDER BY return_rate_pct DESC;

CREATE OR REPLACE VIEW kpi_inventory_turnover AS
SELECT
    i.product_id,
    p.category,
    i.stock_units,
    i.reorder_point,
    COALESCE(SUM(s.quantity), 0) AS units_sold_90d,
    CASE WHEN SUM(s.quantity) > 0
         THEN ROUND(i.stock_units::numeric / SUM(s.quantity), 2)
         ELSE NULL END AS months_of_stock,
    (i.stock_units <= i.reorder_point) AS at_reorder_risk
FROM fact_inventory i
JOIN dim_product p ON i.product_id = p.product_id
LEFT JOIN fact_sales s
    ON i.product_id = s.product_id
   AND s.order_date >= (SELECT MAX(order_date) FROM fact_sales) - INTERVAL '90 days'
GROUP BY 1, 2, 3, 4
ORDER BY months_of_stock ASC NULLS LAST;

CREATE OR REPLACE VIEW kpi_supplier_risk AS
SELECT
    sc.product_id,
    sc.supplier_name,
    sc.reliability_score,
    sc.lead_time_days,
    sc.min_order_qty,
    sc.unit_cost_usd,
    i.stock_units,
    i.reorder_point,
    CASE
        WHEN sc.reliability_score < 0.80 AND i.stock_units <= i.reorder_point THEN 'HIGH'
        WHEN sc.reliability_score < 0.85 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS supplier_risk_level
FROM fact_supplier_costs sc
LEFT JOIN fact_inventory i ON sc.product_id = i.product_id
WHERE sc.is_primary = true;

CREATE OR REPLACE VIEW kpi_marketing_roi AS
SELECT
    year_month,
    channel,
    spend_usd,
    actual_revenue_usd,
    roas,
    cac_usd,
    ROUND((actual_revenue_usd / NULLIF(spend_usd, 0))::numeric, 2) AS revenue_per_spend
FROM fact_marketing_spend
ORDER BY year_month, channel;

CREATE OR REPLACE VIEW kpi_customer_cohort AS
SELECT
    DATE_TRUNC('month', c.registration_date) AS cohort_month,
    c.is_premium,
    COUNT(DISTINCT c.customer_id) AS customers,
    COUNT(DISTINCT s.transaction_id) AS orders,
    ROUND(SUM(s.revenue_usd)::numeric, 2) AS revenue
FROM dim_customer c
LEFT JOIN fact_sales s ON c.customer_id = s.customer_id
                      AND (s.status = 'completed' OR s.status IS NULL)
GROUP BY 1, 2
ORDER BY 1;

-- Regional revenue drill-down, used heavily by the Investigation Agent
CREATE OR REPLACE VIEW kpi_revenue_by_country_month AS
SELECT
    country,
    DATE_TRUNC('month', order_date) AS month,
    SUM(revenue_usd) AS revenue,
    SUM(profit_usd) AS profit,
    COUNT(DISTINCT transaction_id) AS orders
FROM fact_sales
WHERE status = 'completed' OR status IS NULL
GROUP BY 1, 2
ORDER BY 1, 2;

GRANT SELECT ON kpi_revenue_by_month, kpi_returns_rate_by_category, kpi_inventory_turnover,
                kpi_supplier_risk, kpi_marketing_roi, kpi_customer_cohort,
                kpi_revenue_by_country_month
    TO nexus_readonly;
