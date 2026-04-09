USE ROLE ACCOUNTADMIN;
USE ROLE FORECAST_ADMIN_ROLE;
USE WAREHOUSE FORECAST_WH;
USE DATABASE FORECAST_DB;

-- ------------------------------------------------------------
-- Clean up existing objects (if any)
-- ------------------------------------------------------------

-- Drop the warehouse if it exists
-- DROP WAREHOUSE IF EXISTS forecast_wh;

-- Drop the database and everything inside it if it exists
-- DROP DATABASE IF EXISTS forecast_db;

-- Drop roles if they exist
-- DROP ROLE IF EXISTS forecast_admin_role;
-- DROP ROLE IF EXISTS forecast_viewer_role;

-- ------------------------------------------------------------
-- ASSESS UPLOADS
-- ------------------------------------------------------------
USE SCHEMA FORECAST_DB.RAW;
SHOW TABLES IN SCHEMA FORECAST_DB.RAW;
--DROP TABLE forecast_db.raw.client_long;
--DROP TABLE forecast_db.raw.fred_long;
--DROP TABLE forecast_db.raw.forward_long;
--DELETE FROM forecast_db.raw.client_long;
--DELETE FROM forecast_db.raw.fred_long;
--DELETE FROM forecast_db.raw.forward_long;
SELECT * FROM forecast_db.raw.forward_long;
SELECT * FROM forecast_db.raw.fred_long;
SELECT * FROM forecast_db.raw.client_long;


USE SCHEMA FORECAST_DB.META;
SHOW TABLES IN SCHEMA FORECAST_DB.META;
DROP TABLE forecast_db.meta.forecast_registry;
DROP TABLE forecast_db.meta.chart_registry;
DROP TABLE forecast_db.meta.meta_registry;


--DELETE FROM forecast_db.meta.forecast_registry;
SELECT * FROM forecast_db.meta.forecast_registry;
SELECT * FROM forecast_db.meta.chart_registry;
SELECT * FROM forecast_db.meta.meta_registry;

USE SCHEMA OUTPUT;
SHOW TABLES IN SCHEMA FORECAST_DB.OUTPUT;
DROP TABLE forecast_db.output.yoy_trend;
DROP TABLE forecast_db.output.level_actual_fit_forecast;
DROP TABLE forecast_db.output.yoy_actual_fit_forecast;
DROP TABLE forecast_db.output.betas;
DROP TABLE forecast_db.output.forecast_distributions;
DROP TABLE forecast_db.output.price_optimization_table;
DROP TABLE forecast_db.output.holdout_results;
DROP TABLE forecast_db.output.mape_results;

--DELETE FROM forecast_db.output.forecast_results;
--DELETE FROM forecast_db.output.price_optimization_chart;
--DELETE FROM forecast_db.output.price_optimization_table;

SELECT * FROM forecast_db.output.forecast_results;
SELECT * FROM forecast_db.output.price_optimization_chart;
SELECT * FROM forecast_db.output.price_optimization_table;
SELECT * FROM forecast_db.output.units_betas;
SELECT * FROM forecast_db.output.holdout_results;
SELECT * FROM forecast_db.output.mape_results;
SELECT * FROM forecast_db.output.forecast_distributions;










