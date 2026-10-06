-- Metabase keeps its own application state in a separate database on the same server.
-- Run once, as the owner role:  psql -U nexus_user -d postgres -f metabase_init.sql
SELECT 'CREATE DATABASE metabase_app OWNER nexus_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metabase_app')\gexec
