-- ================================================================
-- CONTEXT
-- ================================================================
USE WAREHOUSE FORECAST_WH;
USE DATABASE FORECAST_DB;
USE ROLE FORECAST_ADMIN_ROLE;

-- ================================================================
-- RUN REGISTRY (inspection only)
-- ================================================================
USE SCHEMA FORECAST_DB.META;

-- Show all runs for inspection / debugging
SELECT
    run_id,
    publish_date,
    status,
    is_visible
FROM meta.forecast_registry
ORDER BY publish_date DESC;

-- ================================================================
-- DEFAULT RUN POINTER (LATEST PUBLISHED RUN)
-- ================================================================

CREATE OR REPLACE VIEW meta.v_latest_published_run AS
SELECT
    run_id,
    publish_date
FROM meta.forecast_registry
WHERE
    status = 'published'
    AND is_visible = TRUE
QUALIFY ROW_NUMBER() OVER (ORDER BY publish_date DESC) = 1;

-- sanity check
SELECT * FROM meta.v_latest_published_run;

-- ================================================================
-- OUTPUT SCHEMA
-- ================================================================
USE SCHEMA FORECAST_DB.OUTPUT;

-- ================================================================
-- CANONICAL (BASE) VIEWS
-- NOTE:
--   These views apply NO run_id filtering.
--   They represent the full, immutable history.
-- ================================================================
CREATE OR REPLACE VIEW output.v_yoy_trend_base AS
SELECT *
FROM output.yoy_trend;

CREATE OR REPLACE VIEW output.v_level_actual_fit_forecast_base AS
SELECT *
FROM output.level_actual_fit_forecast;

CREATE OR REPLACE VIEW output.v_yoy_actual_fit_forecast_base AS
SELECT *
FROM output.yoy_actual_fit_forecast;

CREATE OR REPLACE VIEW output.v_price_optimization_table_base AS
SELECT *
FROM output.price_optimization_table;

CREATE OR REPLACE VIEW output.v_holdout_results_base AS
SELECT *
FROM output.holdout_results;

CREATE OR REPLACE VIEW output.v_mape_results_base AS
SELECT *
FROM output.mape_results;

CREATE OR REPLACE VIEW output.v_forecast_distributions_base AS
SELECT *
FROM output.forecast_distributions;

CREATE OR REPLACE VIEW output.v_betas_base AS
SELECT *
FROM output.betas;

-- ================================================================
-- DEFAULT (LATEST-RUN) VIEWS
-- NOTE:
--   These views bind canonical data to the latest published run.
--   They are what the app uses by default.
-- ================================================================
CREATE OR REPLACE VIEW output.v_yoy_trend AS
SELECT
    b.*
FROM output.v_yoy_trend b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_level_actual_fit_forecast AS
SELECT
    b.*
FROM output.v_level_actual_fit_forecast_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

select * from output.v_level_actual_fit_forecast

CREATE OR REPLACE VIEW output.v_yoy_actual_fit_forecast AS
SELECT
    b.*
FROM output.v_yoy_actual_fit_forecast_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_price_optimization_table AS
SELECT
    b.*
FROM output.v_price_optimization_table_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_holdout_results AS
SELECT
    b.*
FROM output.v_holdout_results_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_mape_results AS
SELECT
    b.*
FROM output.v_mape_results_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_forecast_distributions AS
SELECT
    b.*
FROM output.v_forecast_distributions_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW output.v_betas AS
SELECT
    b.*
FROM output.v_betas_base b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id;

-- ================================================================
-- META SCHEMA
-- ================================================================
USE SCHEMA FORECAST_DB.META;

-- ================================================================
-- CANONICAL REGISTRY VIEWS
-- ================================================================

CREATE OR REPLACE VIEW meta.v_chart_registry_base AS
SELECT *
FROM meta.chart_registry;

CREATE OR REPLACE VIEW meta.v_meta_registry_base AS
SELECT *
FROM meta.meta_registry;

-- ================================================================
-- DEFAULT (LATEST-RUN) REGISTRY VIEWS
-- ================================================================

CREATE OR REPLACE VIEW meta.v_chart_registry AS
SELECT
    c.run_id,
    c.publish_date,
    c.chart_key,
    c.chart_group,
    c.chart_template,
    c.model_name,
    c.is_visible
FROM meta.v_chart_registry_base c
JOIN meta.v_latest_published_run r
    ON c.run_id = r.run_id
WHERE
    c.is_visible = TRUE;

CREATE OR REPLACE VIEW meta.v_meta_registry AS
SELECT
    m.run_id,
    m.publish_date,
    m.meta_key,
    m.meta_value,
    m.is_visible
FROM meta.v_meta_registry_base m
JOIN meta.v_latest_published_run r
    ON m.run_id = r.run_id
WHERE
    m.is_visible = TRUE;
