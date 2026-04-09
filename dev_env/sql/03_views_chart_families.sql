USE WAREHOUSE FORECAST_WH;
USE DATABASE FORECAST_DB;
USE ROLE FORECAST_ADMIN_ROLE;
USE SCHEMA FORECAST_DB.OUTPUT;

-----------------------------------------------------------
-- LEVEL ACTUAL / FIT / FORECAST
-----------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_level_actual_fit_forecast AS
SELECT
    lf.run_id,
    lf.series_name,
    lf.period_date,
    lf.value_type,     -- 'actual' | 'fitted' | 'forecast'
    lf.value
FROM output.v_level_actual_fit_forecast lf
JOIN meta.v_latest_published_run r
    ON lf.run_id = r.run_id
ORDER BY
    lf.series_name,
    lf.period_date,
    lf.value_type;

------------------------------------------------------------
-- YOY ACTUAL / FIT / FORECAST
------------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_yoy_actual_fit_forecast AS
SELECT
    yf.run_id,
    yf.series_name,
    yf.period_date,
    yf.value_type,     -- 'actual' | 'fitted' | 'forecast'
    yf.value
FROM output.v_yoy_actual_fit_forecast yf
JOIN meta.v_latest_published_run r
    ON yf.run_id = r.run_id
ORDER BY
    yf.series_name,
    yf.period_date,
    yf.value_type;

------------------------------------------------------------
-- RAW AND NORMALIZED BETAS
------------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_betas AS
SELECT
    b.run_id,
    b.dep,
    b.beta_name,
    b.period_date,
    'raw' AS beta_type,
    b.beta_value
FROM output.v_betas b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id

UNION ALL

SELECT
    b.run_id,
    b.dep,
    b.beta_name,
    b.period_date,
    'normalized' AS beta_type,
    b.beta_value /
        NULLIF(
            SUM(ABS(b.beta_value)) OVER (PARTITION BY b.run_id, b.dep, b.period_date),
            0
        ) AS beta_value
FROM output.v_betas b
JOIN meta.v_latest_published_run r
    ON b.run_id = r.run_id

ORDER BY
    dep,
    beta_name,
    period_date,
    beta_type;

---------------------------------------------------------
-- YOY TRENDS (ACTUAL ONLY)
---------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_yoy_trends AS
SELECT
    yf.run_id,
    yf.series_name,
    yf.period_date,
    yf.value AS yoy_value
FROM output.v_yoy_actual_fit_forecast yf
JOIN meta.v_latest_published_run r
    ON yf.run_id = r.run_id
WHERE yf.value_type = 'actual'
ORDER BY
    yf.series_name,
    yf.period_date;

---------------------------------------------------------
-- FORECAST DISTRIBUTIONS (LEVEL + YOY)
---------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_forecast_distributions AS
SELECT
    fd.run_id,
    fd.dep,
    fd.metric,        -- 'level' | 'yoy'
    fd.bin_id,
    fd.bin_center,
    fd.probability
FROM output.v_forecast_distributions fd
JOIN meta.v_latest_published_run r
    ON fd.run_id = r.run_id
ORDER BY
    fd.dep,
    fd.metric,
    fd.bin_id;

---------------------------------------------------------
-- HOLDOUT PATHS
---------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_holdout_paths AS
SELECT
    ho.run_id,
    ho.dep,
    ho.period_date,
    ho.value_type,    -- 'actual' | 'fit' | 'forecast' | 'ols'
    ho.value
FROM output.v_holdout_results ho
JOIN meta.v_latest_published_run r
    ON ho.run_id = r.run_id
WHERE ho.value_type IN ('actual', 'fit', 'forecast', 'ols')
ORDER BY
    ho.dep,
    ho.period_date,
    ho.value_type;

---------------------------------------------------------
-- MAPE OVER TIME
---------------------------------------------------------
CREATE OR REPLACE VIEW output.v_chart_mape_over_time AS
SELECT
    mr.run_id,
    mr.dep,
    mr.period_date,
    mr.mape_value
FROM output.mape_results mr
JOIN meta.v_latest_published_run r
    ON mr.run_id = r.run_id
ORDER BY
    mr.dep,
    mr.period_date;


SELECT
    value_type,
    MIN(period_date) AS min_dt,
    MAX(period_date) AS max_dt,
    COUNT(value)     AS non_null_cnt
FROM output.v_chart_level_actual_fit_forecast
WHERE series_name = 'units'
GROUP BY value_type
ORDER BY value_type;
