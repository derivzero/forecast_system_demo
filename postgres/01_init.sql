-- ============================================================
-- derivzero Grocery Market Navigator
-- PostgreSQL Initialization Script
-- Mirrors Snowflake FORECAST_DB structure
-- ============================================================

-- ------------------------------------------------------------
-- ROLES
-- ------------------------------------------------------------

CREATE ROLE forecast_admin;
CREATE ROLE forecast_app_role;

-- ------------------------------------------------------------
-- USERS
-- ------------------------------------------------------------

CREATE USER derivzero WITH PASSWORD 'Hambone66!!!';
CREATE USER forecast_app_user WITH PASSWORD 'ScruffyScout!';

GRANT forecast_admin TO derivzero;
GRANT forecast_app_role TO forecast_app_user;

-- ------------------------------------------------------------
-- SCHEMAS
-- ------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS output;
CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS security;

ALTER SCHEMA raw OWNER TO derivzero;
ALTER SCHEMA output OWNER TO derivzero;
ALTER SCHEMA meta OWNER TO derivzero;
ALTER SCHEMA security OWNER TO derivzero;

-- ------------------------------------------------------------
-- SET ROLE - all tables created after SET ROLE is defined are owned by derivzero
-- ------------------------------------------------------------

SET ROLE derivzero;

-- ------------------------------------------------------------
-- RAW TABLES
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS raw.fred_long (
    ingest_date     TIMESTAMP       NOT NULL,
    period_date     DATE            NOT NULL,
    var_name        TEXT,
    var_value       NUMERIC(18,6)
);

CREATE TABLE IF NOT EXISTS raw.client_long (
    ingest_date     TIMESTAMP       NOT NULL,
    period_date     DATE            NOT NULL,
    var_name        TEXT,
    var_value       NUMERIC(18,6)
);

CREATE TABLE IF NOT EXISTS raw.forward_long (
    ingest_date     TIMESTAMP       NOT NULL,
    period_date     DATE            NOT NULL,
    var_name        TEXT,
    var_value       NUMERIC(18,6)
);

-- ------------------------------------------------------------
-- OUTPUT TABLES
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS output.df_wide (
    run_id                          TEXT        NOT NULL,
    period_date                     DATE        NOT NULL,
    avg_cost                        FLOAT,
    avg_cost_fit                    FLOAT,
    avg_cost_fit_yoy                FLOAT,
    avg_cost_forecast               FLOAT,
    avg_cost_forecast_yoy           FLOAT,
    avg_cost_yoy                    FLOAT,
    avg_price                       FLOAT,
    avg_price_log                   FLOAT,
    avg_price_trend                 FLOAT,
    avg_price_trend_yoy             FLOAT,
    avg_price_yoy                   FLOAT,
    cogs                            FLOAT,
    cogs_fit                        FLOAT,
    cogs_fit_yoy                    FLOAT,
    cogs_forecast                   FLOAT,
    cogs_forecast_yoy               FLOAT,
    cogs_yoy                        FLOAT,
    cpi_fah                         FLOAT,
    cpi_fah_fit                     FLOAT,
    cpi_fah_fit_yoy                 FLOAT,
    cpi_fah_forecast                FLOAT,
    cpi_fah_forecast_yoy            FLOAT,
    cpi_fah_yoy                     FLOAT,
    fixed_cost                      FLOAT,
    fixed_cost_yoy                  FLOAT,
    gm                              FLOAT,
    gm_fit                          FLOAT,
    gm_fit_yoy                      FLOAT,
    gm_forecast                     FLOAT,
    gm_forecast_yoy                 FLOAT,
    gm_yoy                          FLOAT,
    home_price                      FLOAT,
    home_price_yoy                  FLOAT,
    net_income                      FLOAT,
    net_income_fit                  FLOAT,
    net_income_fit_yoy              FLOAT,
    net_income_forecast             FLOAT,
    net_income_forecast_yoy         FLOAT,
    net_income_yoy                  FLOAT,
    oil_prices                      FLOAT,
    oil_prices_lag7                 FLOAT,
    oil_prices_lag7_yoy             FLOAT,
    oil_prices_lag8                 FLOAT,
    ppi_farm_products               FLOAT,
    ppi_farm_products_lag4          FLOAT,
    ppi_farm_products_lag4_yoy      FLOAT,
    ppi_farm_products_lag5          FLOAT,
    ppi_food_mfg                    FLOAT,
    ppi_food_mfg_lag2               FLOAT,
    ppi_food_mfg_lag2_log           FLOAT,
    ppi_food_mfg_lag2_yoy           FLOAT,
    ppi_food_mfg_lag3               FLOAT,
    ppi_food_mfg_lag3_yoy           FLOAT,
    ppi_food_mfg_lag4               FLOAT,
    ppi_food_mfg_lag4_log           FLOAT,
    ppi_food_mfg_lag4_yoy           FLOAT,
    ppi_grocery                     FLOAT,
    ppi_grocery_lag1                FLOAT,
    ppi_grocery_yoy                 FLOAT,
    rdi                             FLOAT,
    rdi_yoy                         FLOAT,
    sales                           FLOAT,
    sales_fit                       FLOAT,
    sales_fit_yoy                   FLOAT,
    sales_forecast                  FLOAT,
    sales_forecast_yoy              FLOAT,
    sales_mkt_trend                 FLOAT,
    sales_mkt_trend_fit             FLOAT,
    sales_mkt_trend_fit_yoy         FLOAT,
    sales_mkt_trend_forecast        FLOAT,
    sales_mkt_trend_forecast_yoy    FLOAT,
    sales_mkt_trend_yoy             FLOAT,
    sales_yoy                       FLOAT,
    total_cost                      FLOAT,
    total_cost_fit                  FLOAT,
    total_cost_fit_yoy              FLOAT,
    total_cost_forecast             FLOAT,
    total_cost_forecast_yoy         FLOAT,
    total_cost_yoy                  FLOAT,
    units                           FLOAT,
    units_fit                       FLOAT,
    units_fit_yoy                   FLOAT,
    units_forecast                  FLOAT,
    units_forecast_yoy              FLOAT,
    units_log                       FLOAT,
    units_mkt_trend                 FLOAT,
    units_mkt_trend_fit             FLOAT,
    units_mkt_trend_fit_yoy         FLOAT,
    units_mkt_trend_forecast        FLOAT,
    units_mkt_trend_forecast_yoy    FLOAT,
    units_mkt_trend_yoy             FLOAT,
    units_yoy                       FLOAT,
    upv                             FLOAT,
    upv_fit                         FLOAT,
    upv_fit_yoy                     FLOAT,
    upv_forecast                    FLOAT,
    upv_forecast_yoy                FLOAT,
    upv_yoy                         FLOAT,
    visits                          FLOAT,
    visits_fit                      FLOAT,
    visits_fit_yoy                  FLOAT,
    visits_forecast                 FLOAT,
    visits_forecast_yoy             FLOAT,
    visits_yoy                      FLOAT,
    PRIMARY KEY (run_id, period_date)
);

CREATE TABLE IF NOT EXISTS output.dial_values (
    run_id          TEXT        NOT NULL,
    metric          TEXT,
    value           FLOAT,
    value_type      TEXT
);

CREATE TABLE IF NOT EXISTS output.trees (
    run_id          TEXT        NOT NULL,
    var             TEXT,
    ly              FLOAT,
    ty              FLOAT,
    yoy             FLOAT,
    diff            FLOAT,
    value_type      TEXT
);

CREATE TABLE IF NOT EXISTS output.price_optimization_table (
    run_id          TEXT        NOT NULL,
    avg_price       NUMERIC(18,4),
    units           NUMERIC(18,0)
);

CREATE TABLE IF NOT EXISTS output.betas (
    run_id          TEXT        NOT NULL,
    period_date     DATE        NOT NULL,
    dep             TEXT        NOT NULL,
    beta_name       TEXT        NOT NULL,
    beta_value      FLOAT       NOT NULL
);

CREATE TABLE IF NOT EXISTS output.forecast_distributions (
    run_id          TEXT        NOT NULL,
    dep             TEXT        NOT NULL,
    horizon         TEXT        NOT NULL,
    metric          TEXT        NOT NULL,
    bin_id          NUMERIC(5,0) NOT NULL,
    bin_center      FLOAT       NOT NULL,
    probability     FLOAT       NOT NULL
);

CREATE TABLE IF NOT EXISTS output.holdout_results (
    run_id          TEXT        NOT NULL,
    dep             TEXT        NOT NULL,
    period_date     DATE        NOT NULL,
    value_type      TEXT        NOT NULL,
    value           FLOAT
);

CREATE TABLE IF NOT EXISTS output.mape_results (
    run_id          TEXT        NOT NULL,
    period_date     DATE        NOT NULL,
    mape_value      FLOAT       NOT NULL,
    dep             TEXT        NOT NULL
);

CREATE TABLE IF NOT EXISTS output.chart_artifacts (
    run_id          TEXT        NOT NULL,
    chart_key       TEXT        NOT NULL,
    chart_group     TEXT        NOT NULL,
    chart_type      TEXT        NOT NULL,
    image_data      BYTEA       NOT NULL,
    created_ts      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, chart_key)
);

-- ------------------------------------------------------------
-- META TABLES
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS meta.forecast_registry (
    run_id              TEXT        NOT NULL,
    publish_date        TIMESTAMP   NOT NULL,
    status              TEXT        NOT NULL,
    code_version        TEXT        NOT NULL,
    code_release_date   DATE        NOT NULL,
    is_visible          BOOLEAN     NOT NULL,
    notes               TEXT,
    PRIMARY KEY (run_id)
);

CREATE TABLE IF NOT EXISTS meta.meta_registry (
    run_id          TEXT        NOT NULL,
    publish_date    TIMESTAMP   NOT NULL,
    meta_key        TEXT        NOT NULL,
    meta_value      TEXT,
    PRIMARY KEY (run_id, meta_key)
);

-- ------------------------------------------------------------
-- SECURITY TABLES
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS security.users_access (
    email           TEXT        NOT NULL,
    role            TEXT        NOT NULL DEFAULT 'viewer',
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by      TEXT        NOT NULL DEFAULT CURRENT_USER,
    notes           TEXT,
    last_login_at   TIMESTAMP,
    last_login_ip   TEXT,
    PRIMARY KEY (email)
);

-- ------------------------------------------------------------
-- RESET ROLE - we need to switch back from derivzero to superuser postgres to grant permissions to the other roles
-- ------------------------------------------------------------

RESET ROLE;

-- ------------------------------------------------------------
-- PERMISSIONS - forecast_admin (read and write)
-- Grant of Schema controls who can see inside the schema and run queries against it
-- Grant on all tables controls who can read or write the tables inside the schema
-- ------------------------------------------------------------

GRANT ALL ON SCHEMA raw TO forecast_admin;
GRANT ALL ON SCHEMA output TO forecast_admin;
GRANT ALL ON SCHEMA meta TO forecast_admin;
GRANT USAGE ON SCHEMA security TO forecast_admin;

GRANT ALL ON ALL TABLES IN SCHEMA raw TO forecast_admin;
GRANT ALL ON ALL TABLES IN SCHEMA output TO forecast_admin;
GRANT ALL ON ALL TABLES IN SCHEMA meta TO forecast_admin;
GRANT ALL ON ALL TABLES IN SCHEMA security TO forecast_admin;

-- ------------------------------------------------------------
-- PERMISSIONS - forecast_app_role (read only)
-- ------------------------------------------------------------

GRANT USAGE ON SCHEMA output TO forecast_app_role;
GRANT USAGE ON SCHEMA meta TO forecast_app_role;
GRANT USAGE ON SCHEMA security TO forecast_app_role;

GRANT SELECT ON ALL TABLES IN SCHEMA output TO forecast_app_role;
GRANT SELECT ON ALL TABLES IN SCHEMA meta TO forecast_app_role;
GRANT SELECT ON ALL TABLES IN SCHEMA security TO forecast_app_role;

