from dataclasses import dataclass, field
from datetime import datetime, date
import pandas as pd
from dateutil.relativedelta import relativedelta

@dataclass
class PublishRecord:
    run_id: str                    # "2026_10.0"
    publish_date: datetime         # set at publish time
    status: str                    # "published"
    code_version: str              # "V0", "V1", etc.
    code_release_date: date        # YYYY-MM-DD
    is_visible: bool               # True / False
    notes: str | None = None

@dataclass
class RunConfig:
    # --------------------------------------------------
    # Data windows (ISO date strings, parsed later)
    # --------------------------------------------------
    run_id: str = "2026_04.0"
    
    forecast_month: str = "2026-04-01"
    
    min_date: str = "2014-03-01"

    # Holdout training / test split
    start_train_h: str = field(init=False, default="")
    end_train_h: str   = field(init=False, default="")
    start_test_h: str  = field(init=False, default="")
    end_test_h: str    = field(init=False, default="")

    # Full training window
    start_train: str = field(init=False, default="")
    end_train: str   = field(init=False, default="")

    # Forecast window
    start_fcst: str = field(init=False, default="")
    end_fcst: str   = field(init=False, default="")

    # --------------------------------------------------
    # Chart horizons (years back from forecast end)
    # --------------------------------------------------
    holdout_chart_years: int = 5
    fit_chart_years: int     = 7
    fcst_chart_years: int    = 7
    beta_chart_years: int    = 8
    long_run_chart_years: int = 9

    def __post_init__(self):
        
        # Create relative dates
        forecast_month = pd.to_datetime(self.forecast_month)
        min_date = pd.to_datetime(self.min_date)
        
        self.start_fcst   = forecast_month
        self.end_fcst     = forecast_month + relativedelta(months=5)
        
        self.start_train   = min_date
        self.end_train    = self.start_fcst - relativedelta(months=1)
                
        self.start_train_h = min_date
        self.end_train_h  = self.end_train - relativedelta(months=12)
        
        self.start_test_h = forecast_month - relativedelta(months=12)
        self.end_test_h   = self.end_train

# single shared instance (simple import pattern everywhere)
RUN_CONFIG = RunConfig()

DF_COLUMN_REGISTRY = {
    "period_date": 1,
    "avg_cost": 1,
    "avg_cost_fit": 1,
    "avg_cost_fit_yoy": 1,
    "avg_cost_forecast": 1,
    "avg_cost_forecast_yoy": 1,
    "avg_cost_lag1_log": 0,
    "avg_cost_log": 0,
    "avg_cost_yoy": 1,

    "avg_price": 1,
    "avg_price_gr": 0,
    "avg_price_log": 1,
    "avg_price_log_sq": 0,
    "avg_price_trend": 1,
    "avg_price_trend_log": 0,
    "avg_price_trend_yoy": 1,
    "avg_price_yoy": 1,

    "cogs": 1,
    "cogs_fit": 1,
    "cogs_fit_yoy": 1,
    "cogs_forecast": 1,
    "cogs_forecast_yoy": 1,
    "cogs_yoy": 1,

    "covid1": 0,
    "covid2": 0,

    "cpi": 0,
    "cpi_fah": 1,
    "cpi_fah_100": 0,
    "cpi_fah_fit": 1,
    "cpi_fah_fit_yoy": 1,
    "cpi_fah_forecast": 1,
    "cpi_fah_forecast_yoy": 1,
    "cpi_fah_lag1_log": 0,
    "cpi_fah_log": 0,
    "cpi_fah_yoy": 1,
    "cpi_log": 0,

    "fixed_cost": 1,
    "fixed_cost_yoy": 1,

    "gm": 1,
    "gm_fit": 1,
    "gm_fit_yoy": 1,
    "gm_forecast": 1,
    "gm_forecast_yoy": 1,
    "gm_rate_mo": 0,
    "gm_yoy": 1,

    "home_price": 1,
    "home_price_log": 0,
    "home_price_yoy": 1,

    "import_ffb": 0,

    "m_01": 0,
    "m_02": 0,
    "m_03": 0,
    "m_04": 0,
    "m_05": 0,
    "m_06": 0,
    "m_07": 0,
    "m_08": 0,
    "m_09": 0,
    "m_10": 0,
    "m_11": 0,
    "m_12": 0,

    "net_income": 1,
    "net_income_fit": 1,
    "net_income_fit_yoy": 1,
    "net_income_forecast": 1,
    "net_income_forecast_yoy": 1,
    "net_income_yoy": 1,

    "oil_prices": 1,
    "oil_prices_lag1": 0,
    "oil_prices_lag2": 0,
    "oil_prices_lag3": 0,
    "oil_prices_lag4": 0,
    "oil_prices_lag5": 0,
    "oil_prices_lag6": 0,
    "oil_prices_lag7": 1,
    "oil_prices_lag7_log": 0,
    "oil_prices_lag7_yoy": 1,
    "oil_prices_lag8": 1,
    "oil_prices_lag8_log": 0,
    "oil_prices_lag9": 0,
    "oil_prices_lag9_log": 0,

    "ppi_farm_products": 1,
    "ppi_farm_products_lag1": 0,
    "ppi_farm_products_lag2": 0,
    "ppi_farm_products_lag3": 0,
    "ppi_farm_products_lag4": 1,
    "ppi_farm_products_lag4_log": 0,
    "ppi_farm_products_lag4_yoy": 1,
    "ppi_farm_products_lag5": 1,
    "ppi_farm_products_lag5_log": 0,
    "ppi_farm_products_lag6": 0,
    "ppi_farm_products_lag6_log": 0,
    "ppi_farm_products_lag7": 0,
    "ppi_farm_products_lag7_log": 0,
    "ppi_farm_products_lag8": 0,
    "ppi_farm_products_lag9": 0,

    "ppi_food_mfg": 1,
    "ppi_food_mfg_lag1": 0,
    "ppi_food_mfg_lag2": 1,
    "ppi_food_mfg_lag2_log": 1,
    "ppi_food_mfg_lag2_yoy": 1,
    "ppi_food_mfg_lag3": 1,
    "ppi_food_mfg_lag3_log": 0,
    "ppi_food_mfg_lag3_yoy": 1,
    "ppi_food_mfg_lag4": 1,
    "ppi_food_mfg_lag4_log": 1,
    "ppi_food_mfg_lag4_yoy": 1,
    "ppi_food_mfg_lag5": 0,
    "ppi_food_mfg_lag5_log": 0,
    "ppi_food_mfg_lag6": 0,
    "ppi_food_mfg_lag7": 0,
    "ppi_food_mfg_lag8": 0,
    "ppi_food_mfg_lag9": 0,

    "ppi_grocery": 1,
    "ppi_grocery_lag1": 1,
    "ppi_grocery_lag1_log": 0,
    "ppi_grocery_lag2": 0,
    "ppi_grocery_lag3": 0,
    "ppi_grocery_lag4": 0,
    "ppi_grocery_lag5": 0,
    "ppi_grocery_lag6": 0,
    "ppi_grocery_lag7": 0,
    "ppi_grocery_lag8": 0,
    "ppi_grocery_lag9": 0,
    "ppi_grocery_log": 0,
    "ppi_grocery_yoy": 1,

    "rdi": 1,
    "rdi_log": 0,
    "rdi_yoy": 1,

    "sales": 1,
    "sales_fit": 1,
    "sales_fit_yoy": 1,
    "sales_forecast": 1,
    "sales_forecast_yoy": 1,
    "sales_mkt": 0,
    "sales_mkt_trend": 1,
    "sales_mkt_trend_fit": 1,
    "sales_mkt_trend_fit_yoy": 1,
    "sales_mkt_trend_forecast": 1,
    "sales_mkt_trend_forecast_yoy": 1,
    "sales_mkt_trend_lag1_log": 0,
    "sales_mkt_trend_log": 0,
    "sales_mkt_trend_yoy": 1,
    "sales_yoy": 1,

    "total_cost": 1,
    "total_cost_fit": 1,
    "total_cost_fit_yoy": 1,
    "total_cost_forecast": 1,
    "total_cost_forecast_yoy": 1,
    "total_cost_yoy": 1,

    "units": 1,
    "units_fit": 1,
    "units_fit_yoy": 1,
    "units_forecast": 1,
    "units_forecast_yoy": 1,
    "units_lag1_log": 0,
    "units_log": 1,
    "units_mkt_trend": 1,
    "units_mkt_trend_fit": 1,
    "units_mkt_trend_fit_yoy": 1,
    "units_mkt_trend_forecast": 1,
    "units_mkt_trend_forecast_yoy": 1,
    "units_mkt_trend_lag1_log": 0,
    "units_mkt_trend_log": 0,
    "units_mkt_trend_yoy": 1,
    "units_yoy": 1,

    "upv": 1,
    "upv_fit": 1,
    "upv_fit_yoy": 1,
    "upv_forecast": 1,
    "upv_forecast_yoy": 1,
    "upv_log": 0,
    "upv_yoy": 1,

    "visits": 1,
    "visits_fit": 1,
    "visits_fit_yoy": 1,
    "visits_forecast": 1,
    "visits_forecast_yoy": 1,
    "visits_lag1_log": 0,
    "visits_log": 0,
    "visits_yoy": 1,
}

CHART_KEYS_REQUIRED = [
    "avg_cost_betas",
    "avg_cost_betas_normalized",
    "avg_cost_forecast_distribution",
    "avg_cost_forecast_distribution_yoy",
    "avg_cost_forecast_holdout",
    "avg_cost_level_forecast",
    "avg_cost_level_forecast_yoy",
    "avg_cost_rolling_6m_mape_holdout",
    "avg_cost_yoy",

    "avg_price_and_gm",
    "avg_price_and_net_income",
    "avg_price_trend_yoy",
    "avg_price_yoy",

    "cogs_forecast_distribution",
    "cogs_forecast_distribution_yoy",
    "cogs_level_forecast",
    "cogs_level_forecast_yoy",
    "cogs_yoy",

    "cpi_fah_betas",
    "cpi_fah_betas_normalized",
    "cpi_fah_forecast_distribution",
    "cpi_fah_forecast_distribution_yoy",
    "cpi_fah_forecast_holdout",
    "cpi_fah_level_forecast",
    "cpi_fah_level_forecast_yoy",
    "cpi_fah_rolling_6m_mape_holdout",
    "cpi_fah_yoy",

    "fixed_cost_yoy",

    "gm_forecast_distribution",
    "gm_forecast_distribution_yoy",
    "gm_level_forecast",
    "gm_level_forecast_yoy",
    "gm_yoy",

    "home_price_yoy",

    "market_share_sales",
    "market_share_units",

    "net_income_forecast_distribution",
    "net_income_forecast_distribution_yoy",
    "net_income_level_forecast",
    "net_income_level_forecast_yoy",
    "net_income_yoy",

    "oil_prices_lag7_yoy",

    "ppi_farm_products_lag4_yoy",
    "ppi_food_mfg_lag3_yoy",
    "ppi_grocery_yoy",

    "rdi_yoy",

    "sales_forecast_distribution",
    "sales_forecast_distribution_yoy",
    "sales_level_forecast",
    "sales_level_forecast_yoy",

    "sales_mkt_trend_forecast_distribution",
    "sales_mkt_trend_forecast_distribution_yoy",
    "sales_mkt_trend_level_forecast",
    "sales_mkt_trend_level_forecast_yoy",
    "sales_mkt_trend_yoy",

    "sales_optimization_plot",
    "sales_yoy",

    "total_cost_forecast_distribution",
    "total_cost_forecast_distribution_yoy",
    "total_cost_level_forecast",
    "total_cost_level_forecast_yoy",
    "total_cost_yoy",

    "unit_optimization_plot",

    "units_betas",
    "units_betas_normalized",
    "units_forecast_distribution",
    "units_forecast_distribution_yoy",
    "units_forecast_holdout",
    "units_level_forecast",
    "units_level_forecast_yoy",

    "units_mkt_trend_betas",
    "units_mkt_trend_betas_normalized",
    "units_mkt_trend_forecast_distribution",
    "units_mkt_trend_forecast_distribution_yoy",
    "units_mkt_trend_forecast_holdout",
    "units_mkt_trend_level_forecast",
    "units_mkt_trend_level_forecast_yoy",
    "units_mkt_trend_residuals_histogram",
    "units_mkt_trend_rolling_6m_mape_holdout",
    "units_mkt_trend_yoy",

    "units_rolling_6m_mape_holdout",
    "units_yoy",

    "upv_forecast_distribution",
    "upv_forecast_distribution_yoy",
    "upv_level_forecast",
    "upv_level_forecast_yoy",
    "upv_yoy",

    "visits_betas",
    "visits_betas_normalized",
    "visits_forecast_distribution",
    "visits_forecast_distribution_yoy",
    "visits_forecast_holdout",
    "visits_level_forecast",
    "visits_level_forecast_yoy",
    "visits_rolling_6m_mape_holdout",
    "visits_yoy",
]

# Import CHART_KEYS_REQUIRED_SET whenever needing to filter chardts.
CHART_KEYS_REQUIRED_SET = set(CHART_KEYS_REQUIRED)