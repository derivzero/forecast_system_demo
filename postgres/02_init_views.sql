-- ============================================================
-- derivzero Grocery Market Navigator
-- PostgreSQL View Initialization Script
-- ============================================================

SET ROLE derivzero;

-- ------------------------------------------------------------
-- OUTPUT VIEWS
-- ------------------------------------------------------------

CREATE OR REPLACE VIEW output.v_tree_ttm AS
SELECT run_id, var, ly, ty, yoy, diff, value_type
FROM output.trees
WHERE value_type = 'ttm';

CREATE OR REPLACE VIEW output.v_tree_fcst AS
SELECT run_id, var, ly, ty, yoy, diff, value_type
FROM output.trees
WHERE value_type = 'fcst';

CREATE OR REPLACE VIEW output.v_dial_ttm_yoy AS
SELECT run_id, metric AS var, value
FROM output.dial_values
WHERE value_type = 'ttm';

CREATE OR REPLACE VIEW output.v_dial_forecast_yoy AS
SELECT run_id, metric AS var, value
FROM output.dial_values
WHERE value_type = 'forecast';

-- ------------------------------------------------------------
-- PERMISSIONS
-- ------------------------------------------------------------

GRANT SELECT ON output.v_tree_ttm TO forecast_app_role;
GRANT SELECT ON output.v_tree_fcst TO forecast_app_role;
GRANT SELECT ON output.v_dial_ttm_yoy TO forecast_app_role;
GRANT SELECT ON output.v_dial_forecast_yoy TO forecast_app_role;