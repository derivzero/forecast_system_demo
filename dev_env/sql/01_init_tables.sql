-- ============================================================
-- Base Tables
-- ============================================================

USE ROLE FORECAST_ADMIN_ROLE;
USE WAREHOUSE FORECAST_WH;
USE DATABASE FORECAST_DB;

-- create stage for uploading files. Stage is a temp or put folder that holds uploads
CREATE STAGE IF NOT EXISTS forecast_db.raw.my_stage;

-- ----------------------------------------------------------
-- INPUT DATA - from client and python
-- ----------------------------------------------------------

-- 1 FRED raw data
CREATE OR REPLACE TABLE forecast_db.raw.fred_long (
    ingest_date     TIMESTAMP_NTZ NOT NULL,
    period_date     DATE NOT NULL,
    var_name        STRING NOT NULL,
    var_value       NUMBER(18,6) NOT NULL    
);

-- 2 Client raw data
CREATE OR REPLACE TABLE forecast_db.raw.client_long (
    ingest_date     TIMESTAMP_NTZ NOT NULL,
    period_date     DATE NOT NULL,
    var_name        STRING NOT NULL,
    var_value       NUMBER(18,6) NOT NULL
);

-- 3 Forward values
CREATE OR REPLACE TABLE forecast_db.raw.forward_long (
    ingest_date     TIMESTAMP_NTZ NOT NULL,
    period_date     DATE NOT NULL,
    var_name        STRING NOT NULL,
    var_value       NUMBER(18,6) NOT NULL
);

-- ----------------------------------------------------------
-- OUTPUT DATA - from python
-- ----------------------------------------------------------
CREATE OR REPLACE TABLE forecast_db.output.yoy_trend (
    run_id          STRING NOT NULL,
    period_date     DATE NOT NULL,
    series_name     STRING NOT NULL,          
    value           FLOAT NOT NULL
);

CREATE OR REPLACE TABLE forecast_db.output.level_actual_fit_forecast (
    run_id          STRING NOT NULL,
    period_date     DATE NOT NULL,
    series_name     STRING NOT NULL,          
    value           FLOAT NOT NULL,
    value_type      STRING NOT NULL
);

CREATE OR REPLACE TABLE forecast_db.output.yoy_actual_fit_forecast (
    run_id          STRING NOT NULL,
    period_date     DATE NOT NULL,
    series_name     STRING NOT NULL,          
    value           FLOAT NOT NULL,
    value_type      STRING NOT NULL
);

-- 2 Optimization results
CREATE OR REPLACE TABLE forecast_db.output.price_optimization_table (
    run_id          STRING,
    price           NUMBER(18,4),
    units           NUMBER(18,0),
    sales           NUMBER(18,2),
    cogs            NUMBER(18,2),
    fixed_cost      NUMBER(18,2),
    total_cost      NUMBER(18,2),
    gm              NUMBER(18,2),
    net_income      NUMBER(18,2)
);

CREATE OR REPLACE TABLE forecast_db.output.betas (
    run_id          STRING NOT NULL,
    period_date     DATE NOT NULL,
    dep             STRING NOT NULL,        -- dep var
    beta_name       STRING NOT NULL,        -- ind vars
    beta_value      FLOAT NOT NULL          -- beta
);

CREATE OR REPLACE TABLE output.forecast_distributions (
    run_id        STRING        NOT NULL,
    dep           STRING        NOT NULL,
    horizon       STRING        NOT NULL,   -- e.g. '6m'
    metric        STRING        NOT NULL,   -- 'level' | 'yoy'
    bin_id        NUMBER(5,0)   NOT NULL,   -- 0-29
    bin_center    FLOAT         NOT NULL,   -- x-axis forecast value
    probability   FLOAT         NOT NULL   -- y-axis prob value
);

CREATE OR REPLACE TABLE output.holdout_results (
    run_id       STRING        NOT NULL,
    dep          STRING        NOT NULL,
    period_date  DATE          NOT NULL,
    value_type   STRING        NOT NULL,   -- 'actual' | 'fit' | 'forecast' | 'ols'
    value        FLOAT         
);

CREATE OR REPLACE TABLE output.mape_results (
    run_id       STRING        NOT NULL,
    period_date  DATE          NOT NULL,
    mape_value   FLOAT         NOT NULL,
    dep          STRING        NOT NULL    
);

-- -----------------------------------------------------------
-- 6 META - values are created in Python forecast code
-- ------------------------------------------------------------

CREATE OR REPLACE TABLE meta.forecast_registry (
    run_id              STRING          NOT NULL,  -- YYYY_MM.V, eg 2026_10.0, 2026_10.1
    publish_date        TIMESTAMP_NTZ   NOT NULL,  -- When the publish button was hit YYYY_MM_DD.TIME
    status              STRING          NOT NULL,  -- published | invalid | superseded
    code_version        STRING          NOT NULL,  -- V0, V1, Vn
    code_release_date   DATE            NOT NULL,  -- release date for code version YYYY_MM_DD
    is_visible          BOOLEAN         NOT NULL,  -- (1, 0) boolean           
    notes               STRING,                    -- notes to capture context of run
    PRIMARY KEY (run_id)
);

-- ------------------------------------------------------------
-- the chart and meta registry
-- ------------------------------------------------------------
CREATE OR REPLACE TABLE meta.chart_registry (
    run_id         STRING NOT NULL,
    publish_date   TIMESTAMP_NTZ NOT NULL,
    chart_key      STRING NOT NULL,           -- figure name
    chart_group    STRING NOT NULL,           -- ?
    chart_template STRING NOT NULL,           -- ?
    model_name     STRING,                    -- dep var
    is_visible     BOOLEAN NOT NULL,          -- boolean yes or no
    PRIMARY KEY (run_id, chart_key)
);

CREATE OR REPLACE TABLE meta.meta_registry (
    run_id      STRING NOT NULL,
    publish_date TIMESTAMP_NTZ   NOT NULL,
    meta_key    STRING NOT NULL,
    meta_value  STRING,                 -- keep as STRING to avoid schema churn
    meta_type   STRING,                 -- optional: "float" | "int" | "bool" | "str" | "json"
    is_visible  BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, meta_key)
);



