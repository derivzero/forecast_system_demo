import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)  # For file paths

sys.path.append(project_root)  # For module imports
print(os.getcwd())

# ----------------------------------------------------------
# create the dataset
# ----------------------------------------------------------
def client_data():

    # import files and rename columns
    fred_data = pd.read_csv(
    os.path.join(project_root, "data", "shared", "fetch_fred_data.csv"),
    index_col="date"
    )
    
    fred_data.index = pd.to_datetime(fred_data.index, errors="coerce")
    fred_data = fred_data.sort_index()
    
    fred_data = fred_data.rename(
        columns={
            'CPI':'cpi',
            'CPI (Food at Home)':'cpi_fah', 
            'Real Disposable Income':'rdi', 
            'Avg Home Price':'home_price', 
            'Oil Prices':'oil_prices', 
            'PPI Farm Products':'ppi_farm_products', 
            'PPI Food Manufacture':'ppi_food_mfg', 
            'PPI Grocery':'ppi_grocery',
            'Import Index: Food, Feed, Bev':'import_ffb',
            'Grocery Sales Trend':'sales_mkt_trend',
            'Grocery Sales':'sales'
            }
        )

    # ---------------------------------------------------------
    # Gov't shutdown fix for CPI and CPI FAH which have missing Oct 2025 data
    # ---------------------------------------------------------
    
    fred_data["cpi"] = fred_data["cpi"].interpolate(method="time")
    fred_data["cpi_fah"] = fred_data["cpi_fah"].interpolate(method="time")
    
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
    client_data = pd.merge(fred_data, df_gm_rate_mo, left_index=True, right_index=True, how='left')

    # ---------------------------------------------------------
    # Create Avg Price
    # ---------------------------------------------------------

    # Create avg price data
    client_data["avg_price"] = None

    t = np.arange(len(client_data))
    cpi_fah_base = client_data['cpi_fah'].iloc[0]
    client_data['cpi_fah_100'] = client_data['cpi_fah'] / cpi_fah_base * 100
    client_data['avg_price_gr'] = client_data['cpi_fah_100'] * (1.0001)**t # define the avg_price growth rate by adjusting the cpi_fah index (default = 1.0001)
    
    base_price = 2 # set initial price
    client_data['avg_price_trend'] = (base_price * (client_data['cpi_fah_100']/100))
    client_data['avg_price'] = (base_price * (client_data['avg_price_gr']/100))  # apply the cpi avg price index to base price
    
    # bump sales_trend up to represent the market
    client_data['sales_mkt_trend'] = client_data['sales_mkt_trend'] * 100000
    client_data['units_mkt_trend'] = client_data['sales_mkt_trend']/client_data['avg_price_trend']
    client_data['units'] = client_data['sales']/client_data['avg_price_trend']

    # Create historical unit data
    t = np.arange(len(client_data))
    base_units = client_data['units'].iloc[0] # define the base sales number to create the sales index
    client_data['units_index'] = (client_data['units_mkt_trend'] / base_units) * 100 # create the sales index which is same as cpi_fah above except it is reindexed to 100
    client_data['units_gr'] = client_data['units_index'] * (0.99925)**t  # define the sales growth rate index (default = 0.99925)
    client_data['units_'] = base_units/10 * (client_data['units_gr']/100) # apply adjusted sales index to sales

    # Create some variability in the retailer units
    rng = np.random.default_rng(43)   # reproducible randomness
    sigma = 0.01                       # standard deviation (adjust for how "noisy" you want)
    client_data["unit_rand"] = rng.normal(0, sigma, size=len(client_data))
    client_data["units"] = client_data["units_"] * (1 - client_data["unit_rand"])

    # Create sales, spv, upv, and visits data
    client_data["sales"] = client_data["units"] * client_data["avg_price"]
    client_data["spv"] = (client_data["gm_rate_mo"] * client_data["avg_price"])/2
    client_data["visits"] = client_data["sales"]/client_data["spv"]
    client_data["upv"] = client_data["units"]/client_data["visits"] 

    # ---------------------------------------------------------
    # Create cost and cogs
    # --------------------------------------------------------

    # Create fixed_cost - which is sales * gm_rate as the init value and then grows at the rate of inflation
    cpi_base = client_data['cpi'].iloc[0]
    client_data['cpi_100'] = client_data['cpi'] / cpi_base * 100  
    gm_rate_base = client_data['gm_rate_mo'].iloc[0]
    sales_base = client_data['sales'].iloc[0]
    base_fixed_cost = sales_base * gm_rate_base/100
    client_data['fixed_cost'] = base_fixed_cost * client_data['cpi_100']/100
    client_data['fixed_cost_unit'] = client_data["fixed_cost"]/client_data["units"]
    
    # Define avg_cost - which is price times 1 minus gm rate. It is the gap between price and cost based on the gm rate
    base_cost = base_price * (1 - gm_rate_base/100)
    base_cpi_fah = client_data['cpi_fah'].iloc[0]
    client_data['avg_cost'] = base_cost * (client_data['cpi_fah']/base_cpi_fah)

    # Define total_cost - total_cost per unit time units.  We convert fixed costs into a per unit measure
    client_data['total_cost_unit'] = client_data['avg_cost'] + client_data['fixed_cost_unit']
    client_data['total_cost'] = client_data['total_cost_unit'] * client_data["units"]
            
    # Define COGS - avg_cost time units   
    client_data['cogs'] = client_data['avg_cost'] * client_data['units']

    # Define GM - sales minus cogs 
    client_data["gm"] = client_data['sales'] - client_data['cogs']
    client_data["gm_rate"] = client_data["gm"]/client_data['sales']
    
    # Define Net Income - sales minus total cost
    client_data['net_income'] = client_data['sales'] - client_data['total_cost']
    client_data['net_income'] = client_data['net_income'].replace(0, 2000000)

    # remove missing values
    client_data = client_data.dropna(subset=['sales'])
    client_data = client_data.ffill()

    client_data = client_data[["sales", "units", "visits", "upv", "avg_price", "avg_cost", "fixed_cost", "cogs", "total_cost", "gm", "net_income"]]

    return client_data