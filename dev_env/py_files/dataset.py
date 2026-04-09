import pandas as pd
import numpy as np
import os
import sys
from py_files.model_configs import MODEL_CONFIG as model_cfg
from py_files.run_configs import RUN_CONFIG as run_cfg

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)  # For file paths

sys.path.append(project_root)  # For module imports
print(os.getcwd())

# ----------------------------------------------------------
# create the dataset
# ----------------------------------------------------------
def create_dataset(df_wide, forward_df):

    df_wide = df_wide.copy()
        
    df_wide.index = pd.to_datetime(df_wide.index, errors="coerce")
    df_wide = df_wide.sort_index()
    
    df_wide = df_wide.rename(
        columns={
            'CPI':'cpi',
            'CPI (Food at Home)':'cpi_fah', 
            'Real Disposable Income':'rdi', 
            'Avg Home Price':'home_price', 
            'Oil Prices':'oil_prices', 
            'PPI Farm Products':'ppi_farm_products', 
            'PPI Food Manufacture':'ppi_food_mfg', 
            'PPI Grocery':'ppi_grocery',
            'Grocery Sales Trend':'sales_mkt_trend',
            'Grocery Sales':'sales_mkt'
            }
        )

    # ---------------------------------------------------------
    # Gov't shutdown fix for CPI and CPI FAH which have missing Oct 2025 data
    # ---------------------------------------------------------
    
    df_wide["cpi"] = df_wide["cpi"].interpolate(method="time")
    df_wide["cpi_fah"] = df_wide["cpi_fah"].interpolate(method="time")
        
    # ----------------------------------------------------------
    # Create GM Rate value - Convert Annual GM Rate to monthly value
    # ----------------------------------------------------------
    gm_rate = pd.read_csv(os.path.join(project_root, 'data', "shared", 'gm_rate_annual.csv'))

    gm_rate['year'] = pd.to_datetime(gm_rate['date']).dt.year

    # Convert quarterly to monthly
    gm_rate_mo_rows = [] # holds the values
    rng = np.random.default_rng(42)  # random number generators with seed for reproducibility

    # interate through each annual record/row and extracts the year and the gm rate
    for _, row in gm_rate.dropna(subset=['year', 'gmrate']).iterrows():
        year = int(row['year'])
        total = float(row['gmrate'])

        # figure out first month in the quarter and then build the month labels
        months = [f"{year}-{m:02d}" for m in range(1, 13)]

        # creates size=12 multipliers near loc=1.0 with a stdev(scale) of 0.01. THis adds random variation around the value of 1 or the annual value
        base = rng.normal(loc=1.0, scale=0.01, size=12) 

        base = base / base.mean() # normalize mean to 1 which ensures the 12 weights sum to 1.p
        shrink = 0.5 # 0 = flat and 1 = full wiggle
        weights = 1.0 + shrink * (base - 1.0)
        weights = np.clip(weights, 0.985, 1.015) # clips extreme values to (weights, lower bound, upper bound)

        # loop through the months and weights and append valuess to gm_rate_mo_rows list above.  Creates one row per month
        for m, w in zip(months, weights):
            gm_rate_mo_rows.append({'month': m, 'gm_rate_mo': round(total * w, 4)})

    # final DataFrame
    df_gm_rate_mo = pd.DataFrame(gm_rate_mo_rows)
    df_gm_rate_mo['month'] = pd.to_datetime(df_gm_rate_mo['month'])
    df_gm_rate_mo = df_gm_rate_mo.set_index('month')
    df_gm_rate_mo = df_gm_rate_mo.sort_values('month')

    # merge with fred data
    df_wide = pd.merge(df_wide, df_gm_rate_mo, left_index=True, right_index=True, how='left')

    # ---------------------------------------------------------
    # Create Market Vars
    # ---------------------------------------------------------

    # Create avg price data
    df_wide["avg_price"] = None

    t = np.arange(len(df_wide))
    cpi_fah_base = df_wide['cpi_fah'].iloc[0]
    df_wide['cpi_fah_100'] = df_wide['cpi_fah'] / cpi_fah_base * 100
    df_wide['avg_price_gr'] = df_wide['cpi_fah_100'] * (1.0001)**t # define the avg_price growth rate by adjusting the cpi_fah index (default = 1.0001)
    
    base_price = 2 # set initial price
    df_wide['avg_price_trend'] = (base_price * (df_wide['cpi_fah_100']/100))
    df_wide['avg_price'] = (base_price * (df_wide['avg_price_gr']/100))  # apply the cpi avg price index to base price
    
    # bump sales_trend up to represent the market
    df_wide['sales_mkt_trend'] = df_wide['sales_mkt_trend'] * 100000
    df_wide['units_mkt_trend'] = df_wide['sales_mkt_trend']/df_wide['avg_price_trend']

    # remove missing values
    #df_wide = df_wide.dropna(subset=['sales'])
    df_wide = df_wide.ffill()

    #----------------------------------------------------------------
    # Create lagged variables
    # ----------------------------------------------------------------

    lag_vars = ['oil_prices', 'ppi_farm_products', 'ppi_food_mfg', 'ppi_grocery']

    lags = range(1, 10)

    lagged_cols = {}

    for lag_var in lag_vars:
        for lag in lags:
            col_name = f'{lag_var}_lag{lag}'
            lagged_cols[col_name] = df_wide[lag_var].shift(lag)

    # Merge all lagged columns at once using concat
    df_wide = pd.concat([df_wide, pd.DataFrame(lagged_cols, index=df_wide.index)], axis=1)
    
    # set date as datetime and index    
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide.index.name = "date"

    # bring in rdi adjusted for grocery forecast
    rdi_adj = pd.read_excel(os.path.join(project_root, 'data', 'shared', 'RDI Analysis.xlsx'))
    rdi_adj = rdi_adj[['observation_date', 'RDI_adj']]
    rdi_adj = rdi_adj.rename(columns={'observation_date':'date', 'RDI_adj':'rdi_adj'})
    rdi_adj = rdi_adj.set_index('date')

    # join RDI with Fred Data
    df_wide = df_wide.join(rdi_adj, how='left')
    df_wide["rdi_adj"] = df_wide["rdi_adj"].replace(r"^\s*$", pd.NA, regex=True)
    df_wide['rdi'] = df_wide["rdi_adj"].fillna(df_wide["rdi"])

    # set date as datetime and index    
    df_wide.index = pd.to_datetime(df_wide.index)
    df_wide.index.name = "date"

    df_wide = df_wide.loc[run_cfg.start_train:run_cfg.end_train]

    # --------------------------------------------------------------------------------
    # Convert YOY forward values into actual values
    # --------------------------------------------------------------------------------
    forward_df = forward_df.copy()
    forward_df.index = pd.to_datetime(forward_df.index, errors="coerce")
    forward_df = forward_df.loc[run_cfg.start_fcst:run_cfg.end_fcst]

    value_col = ['cpi', 'fixed_cost', 'rdi', 'home_price', 'avg_price', 'avg_price_trend',
                 "oil_prices_lag7", "oil_prices_lag8", "oil_prices_lag9", 
                 "ppi_farm_products_lag4", "ppi_farm_products_lag5", "ppi_farm_products_lag6", "ppi_farm_products_lag7", 
                 "ppi_food_mfg_lag2", "ppi_food_mfg_lag3", "ppi_food_mfg_lag4", "ppi_food_mfg_lag5", 
                 "ppi_grocery", "ppi_grocery_lag1"]
    
    yoy_col   = ['cpi_yoy', 'fixed_cost_yoy', 'rdi_yoy', 'home_price_yoy', 'avg_price_yoy', 'avg_price_trend_yoy',
                 "oil_prices_yoy_lag7", "oil_prices_yoy_lag8", "oil_prices_yoy_lag9", 
                 "ppi_farm_products_yoy_lag4", "ppi_farm_products_yoy_lag5", "ppi_farm_products_yoy_lag6", "ppi_farm_products_yoy_lag7", 
                 "ppi_food_mfg_yoy_lag2", "ppi_food_mfg_yoy_lag3", "ppi_food_mfg_yoy_lag4", "ppi_food_mfg_yoy_lag5", 
                 "ppi_grocery_yoy", "ppi_grocery_yoy_lag1"] 
    
    for col in value_col:
        forward_df[col] = np.nan
    
    fwd_base = forward_df[value_col]
    df_wide = pd.concat([df_wide, fwd_base], axis=0)
    df_wide = df_wide.sort_index()
  
    fwd_yoy = forward_df[yoy_col] # removes dummy vars

    for val, yoy in zip(value_col, yoy_col):

        # last year's value
        prev_year = df_wide[val].shift(12)

        # forward rows only
        mask = df_wide.index.isin(forward_df.index)

        # fill level value using YOY formula for those rows in the mask filter
        df_wide.loc[mask, val] = (
            prev_year.loc[mask] * (1 + fwd_yoy[yoy])
        )
        
    # -------------------------------------------------------------------
    # Create log variables
    # -------------------------------------------------------------------
    col_logs = ['cpi_fah', 'sales_mkt_trend', 'units_mkt_trend', 
                'units', 'visits', 'avg_cost', 'avg_price', 'upv',
                'cpi', 'rdi', 'home_price', 'avg_price', 'avg_price_trend',
                 "oil_prices_lag7", "oil_prices_lag8", "oil_prices_lag9", 
                 "ppi_farm_products_lag4", "ppi_farm_products_lag5", "ppi_farm_products_lag6", "ppi_farm_products_lag7", 
                 "ppi_food_mfg_lag2", "ppi_food_mfg_lag3", "ppi_food_mfg_lag4", "ppi_food_mfg_lag5", 
                 "ppi_grocery", "ppi_grocery_lag1"]
        
    for col in col_logs:
        df_wide[f"{col}_log"] = np.log(df_wide[col])
   
    # create lagged dependent vars
    df_wide['cpi_fah_lag1_log'] = df_wide['cpi_fah_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'cpi_fah_lag1_log'] = df_wide.loc[df_wide.index[0], 'cpi_fah_log']
    df_wide["sales_mkt_trend_lag1_log"] = df_wide['sales_mkt_trend_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'sales_mkt_trend_lag1_log'] = df_wide.loc[df_wide.index[0], 'sales_mkt_trend_log']
    df_wide["units_mkt_trend_lag1_log"] = df_wide['units_mkt_trend_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'units_mkt_trend_lag1_log'] = df_wide.loc[df_wide.index[0], 'sales_mkt_trend_log']
    df_wide["units_lag1_log"] = df_wide['units_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'units_lag1_log'] = df_wide.loc[df_wide.index[0], 'units_log']
    df_wide["visits_lag1_log"] = df_wide['visits_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'visits_lag1_log'] = df_wide.loc[df_wide.index[0], 'visits_log']
    df_wide["avg_cost_lag1_log"] = df_wide['avg_cost_log'].shift(1)
    df_wide.loc[df_wide.index[0], 'avg_cost_lag1_log'] = df_wide.loc[df_wide.index[0], 'avg_cost_log']

    # ------------------------------------------------------------------
    # Create dummy Vars
    # ------------------------------------------------------------------    
       
    # Create dummy1 for dates between 2/29/2020 and 4/30/2020. A boolean consition with a .astype(int) will return a 1 for True and 0 for false
    df_wide['covid1'] = ((df_wide.index >= '2020-02-01') & (df_wide.index <= '2020-04-01')).astype(int)

    # Create dummy2 for dates between 5/31/2020 and 10/31/2020
    df_wide['covid2'] = ((df_wide.index >= '2020-05-01') & (df_wide.index <= '2020-10-01')).astype(int)

    # month numbers from index
    dummies = pd.get_dummies(df_wide.index.month, dtype=int)
    dummies.columns = [f"m_{int(i):02d}" for i in dummies.columns]
    dummies.index = df_wide.index  # align indices

    df_wide = df_wide.join(dummies)

    # ------------------------------------------------------------------
    # Create nonlinear Vars
    # ------------------------------------------------------------------    
    df_wide['avg_price_log_sq'] = df_wide['avg_price_log'] ** 2

    return df_wide