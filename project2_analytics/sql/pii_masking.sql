-- PII masking: the agent's read-only role sees a view without names or emails,
-- and has no direct access to the tables that hold them.
-- Re-run this after anything that re-applies grants (e.g. schema_star.sql).
CREATE OR REPLACE VIEW dim_customer_safe AS
SELECT customer_id, country, currency, age, gender,
       registration_date, is_premium, email_verified
FROM dim_customer;

GRANT SELECT ON dim_customer_safe TO nexus_readonly;
REVOKE ALL ON dim_customer FROM nexus_readonly;
REVOKE ALL ON raw_customers FROM nexus_readonly;
