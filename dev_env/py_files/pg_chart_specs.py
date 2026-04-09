# sf_charts_rendering.py
import pandas as pd
import numpy as np

CHART_SPECS = {
    "avg_cost_betas": {"chart_group": "betas", "chart_type": "betas", "var_name": "avg_cost"},
    "avg_cost_betas_normalized": {"chart_group": "betas", "chart_type": "betas_normalized", "var_name": "avg_cost"},
    "avg_cost_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "avg_cost"},
    "avg_cost_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "avg_cost"},
    "avg_cost_forecast_holdout": {"chart_group": "performance", "chart_type": "forecast_holdout", "var_name": "avg_cost"},
    "avg_cost_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "avg_cost"},
    "avg_cost_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "avg_cost"},
    "avg_cost_rolling_6m_mape_holdout": {"chart_group": "performance", "chart_type": "rolling_6m_mape", "var_name": "avg_cost"},
    "avg_cost_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "avg_cost"},
    "avg_cost_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "avg_cost"},
        
    "avg_price_and_gm": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "avg_price_and_gm"},
    "avg_price_and_net_income": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "avg_price_and_net_income"},
    "avg_price_trend_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "avg_price_trend"},
    "avg_price_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "avg_price"},
    "avg_price_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "avg_price"},
    
    "cogs_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "cogs"},
    "cogs_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "cogs"},
    "cogs_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "cogs"},
    "cogs_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "cogs"},
    "cogs_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "cogs"},
    "cogs_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "cogs"},
        
    "cpi_fah_betas": {"chart_group": "betas", "chart_type": "betas", "var_name": "cpi_fah"},
    "cpi_fah_betas_normalized": {"chart_group": "betas", "chart_type": "betas_normalized", "var_name": "cpi_fah"},
    "cpi_fah_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "cpi_fah"},
    "cpi_fah_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "cpi_fah"},
    "cpi_fah_forecast_holdout": {"chart_group": "performance", "chart_type": "forecast_holdout", "var_name": "cpi_fah"},
    "cpi_fah_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "cpi_fah"},
    "cpi_fah_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "cpi_fah"},
    "cpi_fah_rolling_6m_mape_holdout": {"chart_group": "performance", "chart_type": "rolling_6m_mape", "var_name": "cpi_fah"},
    "cpi_fah_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "cpi_fah"},
        
    "fixed_cost_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "fixed_cost"},
    
    "gm_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "gm"},
    "gm_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "gm"},
    "gm_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "gm"},
    "gm_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "gm"},
    "gm_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "gm"},
    "gm_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "gm"},
        
    "home_price_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "home_price"},
    
    "market_share_sales": {"chart_group": "optimization", "chart_type": "market_share", "var_name": "market_share_sales"},
    "market_share_units": {"chart_group": "optimization", "chart_type": "market_share", "var_name": "market_share_units"},

    "net_income_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "net_income"},
    "net_income_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "net_income"},
    "net_income_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "net_income"},
    "net_income_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "net_income"},
    "net_income_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "net_income"},
    
    "oil_prices_lag7_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "oil_prices_lag7"},

    "ppi_farm_products_lag3_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_farm_products_lag3"},
    "ppi_farm_products_lag4_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_farm_products_lag4"},
    "ppi_farm_products_lag5_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_farm_products_lag5"},
    
    "ppi_food_mfg_lag2_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_food_mfg_lag2"},
    "ppi_food_mfg_lag3_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_food_mfg_lag3"},
    "ppi_food_mfg_lag4_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_food_mfg_lag4"},
    
    "ppi_grocery_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "ppi_grocery"},
    "price_range" : {"chart_group": "optimization", "chart_type": "price_range", "var_name": "units"},
        
    "rdi_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "rdi"},
    
    "sales_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "sales"},
    "sales_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "sales"},
    "sales_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "sales"},
    "sales_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "sales"},
    "sales_price_min": {"chart_group": "optimization", "chart_type": "min_price", "var_name": "sales"},
    "sales_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "sales"},
    "sales_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "sales"},
    
    "sales_mkt_trend_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "sales_mkt_trend"},
    "sales_mkt_trend_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "sales_mkt_trend"},
    "sales_mkt_trend_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "sales_mkt_trend"},
    "sales_mkt_trend_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "sales_mkt_trend"},
    "sales_mkt_trend_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "sales_mkt_trend"},
        
    "total_cost_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "total_cost"},
    "total_cost_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "total_cost"},
    "total_cost_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "total_cost"},
    "total_cost_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "total_cost"},
    "total_cost_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "total_cost"},
    
    "units_betas": {"chart_group": "betas", "chart_type": "betas", "var_name": "units"},
    "units_betas_normalized": {"chart_group": "betas", "chart_type": "betas_normalized", "var_name": "units"},
    "units_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "units"},
    "units_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "units"},
    "units_forecast_holdout": {"chart_group": "performance", "chart_type": "forecast_holdout", "var_name": "units"},
    "units_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "units"},
    "units_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "units"},
    "units_rolling_6m_mape_holdout": {"chart_group": "performance", "chart_type": "rolling_6m_mape", "var_name": "units"},
    "units_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "units"},
    "units_price_max": {"chart_group": "optimization", "chart_type": "max_price", "var_name": "units"},
    "units_ttm_comp": {"chart_group": "ttm_comp", "chart_type": "ttm_comp_time_series", "var_name": "units"},
    
    "units_mkt_trend_betas": {"chart_group": "betas", "chart_type": "betas", "var_name": "units_mkt_trend"},
    "units_mkt_trend_betas_normalized": {"chart_group": "betas", "chart_type": "betas_normalized", "var_name": "units_mkt_trend"},
    "units_mkt_trend_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "units_mkt_trend"},
    "units_mkt_trend_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "units_mkt_trend"},
    "units_mkt_trend_forecast_holdout": {"chart_group": "performance", "chart_type": "forecast_holdout", "var_name": "units_mkt_trend"},
    "units_mkt_trend_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "units_mkt_trend"},
    "units_mkt_trend_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "units_mkt_trend"},
    "units_mkt_trend_rolling_6m_mape_holdout": {"chart_group": "performance", "chart_type": "rolling_6m_mape", "var_name": "units_mkt_trend"},
    "units_mkt_trend_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "units_mkt_trend"},
        
    "upv_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "upv"},
    "upv_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "upv"},
    "upv_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "upv"},
    "upv_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "upv"},
    "upv_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "upv"},
    
    "visits_betas": {"chart_group": "betas", "chart_type": "betas", "var_name": "visits"},
    "visits_betas_normalized": {"chart_group": "betas", "chart_type": "betas_normalized", "var_name": "visits"},
    "visits_forecast_distribution": {"chart_group": "forecast", "chart_type": "forecast_distribution", "var_name": "visits"},
    "visits_forecast_distribution_yoy": {"chart_group": "forecast", "chart_type": "forecast_distribution_yoy", "var_name": "visits"},
    "visits_forecast_holdout": {"chart_group": "performance", "chart_type": "forecast_holdout", "var_name": "visits"},
    "visits_level_forecast": {"chart_group": "forecast", "chart_type": "level_forecast", "var_name": "visits"},
    "visits_level_forecast_yoy": {"chart_group": "forecast", "chart_type": "level_forecast_yoy", "var_name": "visits"},
    "visits_rolling_6m_mape_holdout": {"chart_group": "performance", "chart_type": "rolling_6m_mape", "var_name": "visits"},
    "visits_yoy": {"chart_group": "yoy_trend", "chart_type": "yoy_time_series", "var_name": "visits"},
}






