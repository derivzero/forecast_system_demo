from dataclasses import dataclass
import pandas as pd
from typing import Optional, Callable

# ModelConfig is the blank form or blueprint.  It defines the structure and default values for model configurations.
# This is all of the content above MODEL_CONFIGS
# this is a decorator that automatically adds special methods to the class.

@dataclass 
class ModelConfig: 
    name: str
    type: str
    dep: str
    yoy_charts: list[str]
    beta_charts: list[str]
    agg_rule: str
    dep_log: Optional[str] = None
    lag_log: Optional[str] = None
    ind: Optional[list[str]] = None
    ind_log: Optional[list[str]] = None
    q: Optional[float] = None
    r: Optional[float] = None
    sim_func: Optional[Callable] = None

    # flags
    cpi_special: bool = False

# ------------------------------------------------------------
# FULL MODEL CONFIG DICTIONARY — This is NOT part of the class.  It is a dictionary that will be used to loop through the forecast models.
# Keys must be exact when referencing in the code
# ------------------------------------------------------------
MODEL_CONFIG = {

    "cpi_fah": ModelConfig(
        name="cpi_fah",
        type="stat_model",
        dep="cpi_fah",
        ind=["oil_prices_lag7","oil_prices_lag8",
             "ppi_farm_products_lag4","ppi_farm_products_lag5",
             "ppi_food_mfg_lag3","ppi_food_mfg_lag4",
             "ppi_grocery","ppi_grocery_lag1"],
        dep_log="cpi_fah_log",
        lag_log="cpi_fah_lag1_log",
        ind_log=["cpi_fah_lag1_log",
                "oil_prices_lag7_log","oil_prices_lag8_log",
                "ppi_farm_products_lag4_log","ppi_farm_products_lag5_log",
                "ppi_food_mfg_lag3_log","ppi_food_mfg_lag4_log",
                "ppi_grocery_log","ppi_grocery_lag1_log"],
        yoy_charts=["cpi_fah_yoy", 'oil_prices_lag7_yoy', 'ppi_farm_products_lag4_yoy', 'ppi_food_mfg_lag3_yoy', 'ppi_grocery_yoy'],
        beta_charts=['oil_prices_lag7_log', 'ppi_farm_products_lag4_log', 'ppi_food_mfg_lag3_log', 'ppi_grocery_log'],
        agg_rule="mean",
    

        # KF tuning
        q=1e-8,
        r=5e-3,

        # Process flags
        cpi_special=True
    ),
    
    "units_mkt_trend": ModelConfig(
        name="units_mkt_trend",
        type="stat_model",
        dep="units_mkt_trend",
        ind=["avg_price_trend", "rdi", "home_price"],
        dep_log="units_mkt_trend_log",
        lag_log="units_mkt_trend_lag1_log",
        ind_log=["units_mkt_trend_lag1_log", 'avg_price_trend_log', 'rdi_log', 'home_price_log', "covid1", "covid2"],
        yoy_charts=["units_mkt_trend_yoy", 'avg_price_trend_yoy', 'rdi_yoy', 'home_price_yoy'],
        beta_charts=["units_mkt_trend_lag1_log", 'avg_price_trend_log', 'rdi_log', 'home_price_log'],
        agg_rule="sum",
        
        # Optimal 202510
        #q=0.00000001,
        #r=1

        # KF Tuning
        q=0.00000001,
        r=1
    ),

    "units": ModelConfig(
        name="units",
        type="stat_model",
        dep="units",
        ind=["avg_price", "rdi", "home_price",
             'covid1', 'covid2', 'm_01','m_03','m_04','m_05','m_06','m_07','m_08','m_09','m_10','m_11','m_12'],
        dep_log="units_log",
        lag_log="units_lag1_log",
        ind_log=["units_lag1_log",'avg_price_log', 'rdi_log', 'home_price_log', 
        'covid1', 'covid2', 'm_01','m_03','m_04','m_05','m_06','m_07','m_08','m_09','m_10','m_11','m_12'],
        yoy_charts=['units_yoy', 'avg_price_yoy', 'rdi_yoy', 'home_price_yoy'],
        beta_charts=["units_lag1_log", 'avg_price_log', 'rdi_log', 'home_price_log'],
        agg_rule="sum",
        
        # Optimal 202510
        #q=0.00000001,
        #r=1        
        
        # KF Tuning
        q=.00000001,
        r=1, 
    ),

    "visits": ModelConfig(
        name="visits",
        type="stat_model",
        dep="visits",
        ind=["avg_price", "units"],
        dep_log="visits_log",
        lag_log="visits_lag1_log",
        ind_log=["visits_lag1_log", 'avg_price_log', 'units_log', 'covid1', 'covid2', 
            'm_01','m_03','m_04','m_05','m_06','m_07','m_08','m_09','m_10','m_11','m_12'],
        yoy_charts=['visits_yoy', 'avg_price_yoy', 'units_yoy'],
        beta_charts=['visits_lag1_log', 'avg_price_log', 'units_log'],
        agg_rule="sum",
        
        # KF Tuning
        # q=.5,
        # r=1,
        
        # KF Tuning
        q=.5,
        r=1,
    ),

    "avg_cost": ModelConfig(
        name="avg_cost",
        type="stat_model",
        dep="avg_cost",
        ind=["ppi_food_mfg_lag2", "ppi_food_mfg_lag3", "ppi_food_mfg_lag4"],
        dep_log="avg_cost_log",
        lag_log="avg_cost_lag1_log",
        ind_log=["avg_cost_lag1_log", "ppi_food_mfg_lag2_log", "ppi_food_mfg_lag3_log", "ppi_food_mfg_lag4_log", 'covid1', 'covid2'],
        yoy_charts=["avg_cost_yoy", 'ppi_food_mfg_lag2_yoy', 'ppi_food_mfg_lag3_yoy', 'ppi_food_mfg_lag4_yoy'],
        beta_charts=["avg_cost_lag1_log", 'ppi_food_mfg_lag2_log', 'ppi_food_mfg_lag3_log', 'ppi_food_mfg_lag4_log'],
        agg_rule="mean",

        # KF Tuning
        q=1e-8,
        r=.001
    ),

    "sales": ModelConfig(
        name = "sales",
        type = "derived",
        sim_func=lambda units_sims, df_future: units_sims * df_future["avg_price"].values.reshape(1, -1),
        dep = "sales",
        ind = ["units", "avg_price"],
        yoy_charts=['sales_yoy', 'units_yoy', 'avg_price_yoy'],
        beta_charts = None,
        agg_rule="sum",
    ),

    "sales_mkt_trend": ModelConfig(
        name = "sales_mkt_trend",
        type = "derived",
        sim_func=lambda units_mkt_trend_sims, df_future: units_mkt_trend_sims * df_future["avg_price_trend"].values.reshape(1, -1),
        dep = "sales_mkt_trend",
        ind = ["units_mkt_trend", "avg_price_trend"],
        yoy_charts=["sales_mkt_trend_yoy", 'units_mkt_trend_yoy', "avg_price_trend_yoy"],
        beta_charts = None,
        agg_rule="sum",
    ),

    "cogs": ModelConfig(
        name = "cogs",
        type = "derived",
        sim_func = lambda units_sims, avg_cost_sims: units_sims * avg_cost_sims,
        dep = "cogs",
        ind = ["units", "avg_cost"],
        yoy_charts=['cogs_yoy', 'units_yoy', 'avg_cost_yoy'],
        beta_charts = None,
        agg_rule="sum",
    ),

    "gm": ModelConfig(
        name = "gm",
        type = "derived",
        sim_func = lambda sales_sims, cogs_sims: sales_sims - cogs_sims,
        dep = "gm",
        ind = ["sales", "cogs"],
        yoy_charts=['gm_yoy', 'sales_yoy', 'cogs_yoy'],
        beta_charts = None,
        agg_rule="sum",
    ),

    "total_cost": ModelConfig(
        name = "total_cost",
        type = "derived",
        sim_func = lambda cogs_sims, df_future: cogs_sims + df_future["fixed_cost"].values.reshape(1, -1),
        dep = "total_cost",
        ind = ["cogs", "fixed_cost"],
        yoy_charts=['total_cost_yoy', 'cogs_yoy', 'fixed_cost_yoy'],
        beta_charts = None,
        agg_rule="sum",
    ),

    "net_income": ModelConfig(
        name = "net_income",
        type = "derived",
        sim_func = lambda sales_sims, total_cost_sims: sales_sims - total_cost_sims,
        dep = "net_income",
        ind = ["sales", "total_cost"],
        yoy_charts=['net_income_yoy', 'sales_yoy', 'total_cost_yoy'],
        beta_charts = None,
        agg_rule="sum",
    ),

    "upv": ModelConfig(
        name = "upv",
        type = "derived",
        sim_func = lambda units_sims, visits_sims: units_sims / visits_sims,
        dep = "upv",
        ind = ["units", "visits"],
        yoy_charts=['upv_yoy', 'units_yoy', 'visits_yoy'],
        beta_charts = None,
        agg_rule="mean",
    ),
}





 