def run_optimization(df, cfg_units, kf_results_units, elasticities):
        
    from py_files.charts_helpers import plot_price_curve, plot_units_vs_price, plot_price_range, scatter_by_year, market_share_by_year
    from py_files.optimize_helpers import create_price_grid_df
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import os, json
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from datetime import datetime
    from py_files.run_configs import RUN_CONFIG as cfg_run

    df = df.copy()

    meta = {}
    charts = {}

    # create optimization grids for both linear and nonlinear chart -------------------------------
    
    # avg_price grid for linear chart
    price_grid_df = create_price_grid_df(df, cfg_units, kf_results_units)
    
    # ---------------------------------------------------------------
    # Elasticity Curve Across All Price Points
    # ---------------------------------------------------------------
    
    # calculated in master.py
    E_rel = elasticities["E_rel"]
    E_cat = elasticities["E_cat"]

    meta["units_E_cat"] = elasticities["E_cat"]
    meta["units_E_rel"] = elasticities["E_rel"]

    assert E_rel < 0, "E_rel should be negative"
    assert E_cat <= 0, "E_cat should be non-positive"
    
    # -------------------------------------------------------------------
    # Sales Break Even
    # -------------------------------------------------------------------

    df_fcst = df.loc[cfg_run.start_fcst: cfg_run.end_fcst]
    exp_price = round(df_fcst['sales'].sum() / df_fcst['units'].sum(), 2)

    meta["expected_price_6m"] = float(exp_price)

    fcst_idx = df.loc[cfg_run.start_fcst : cfg_run.end_fcst].index
    ly_idx = fcst_idx - pd.DateOffset(months=12)
    sales_ly = df.loc[ly_idx, 'sales'].sum()

    # find avg_price where projected sales are closest to last year's sales (sales break even)
    diff = (price_grid_df['units'] * price_grid_df['avg_price'] - sales_ly).abs()
    i = diff.idxmin()
    closest_row   = price_grid_df.loc[i]
    sales_breakeven_price = float(closest_row['avg_price'])

    meta["sales_breakeven_price"] = sales_breakeven_price

    price_grid_df["sales"] = price_grid_df["avg_price"] * price_grid_df["units"]
    fig = plot_price_curve(df=price_grid_df, exp_price=exp_price, opt_price=sales_breakeven_price, var='sales', title="Sales Break Even")
    charts["sales_price_min"] = fig
    plt.close(fig)

    # -------------------------------------------------------------------
    # Units Break Even
    # -------------------------------------------------------------------

    # Define indices
    fcst_idx = df.loc[cfg_run.start_fcst : cfg_run.end_fcst].index
    ly_idx = fcst_idx - pd.DateOffset(months=12)

    # Define forecast values from last year for the unit break even
    units_ly = df.loc[ly_idx, 'units'].sum()

    # find avg_price where projected units are closest to last year's units (units break even)
    diff = (price_grid_df['units'] - units_ly).abs()
    i = diff.idxmin()
    closest_row   = price_grid_df.loc[i]
    units_breakeven_price = float(closest_row['avg_price'])

    meta["units_breakeven_price"] = units_breakeven_price

    fig = plot_units_vs_price(df=price_grid_df, exp_price=exp_price, units_opt_price=units_breakeven_price, title="Units Break Even")
    charts["units_price_max"] = fig
    plt.close(fig)

    fig = plot_price_range(sales_breakeven_price, units_breakeven_price, exp_price)
    charts["price_range"] = fig
    plt.close(fig)

    # Show long run charts ---------------------------------------------
    start_date  = cfg_run.end_train - pd.DateOffset(years=cfg_run.long_run_chart_years)  
    end_date = cfg_run.end_train
    df_chart = df.loc[start_date:end_date, ['avg_price', 'gm', 'net_income']]

    fig = scatter_by_year(x=df_chart['avg_price'], y=df_chart['gm'], x_label="Avg Price", y_label="Gross Margin Dollars", title="Average Price and Gross Margin Dollars over Time")
    charts["avg_price_and_gm"] = fig
    plt.close(fig)

    fig = scatter_by_year(x=df_chart['avg_price'], y=df_chart['net_income'], x_label="Avg Price", y_label="Net Income", title="Average Price and Net Income over Time")
    charts["avg_price_and_net_income"] = fig
    plt.close(fig)
    
    # Market Share ----------------------------------------------------
    df['market_share_sales'] = df["sales"]/df["sales_mkt_trend"]
    df['market_share_units'] = df["units"]/df["units_mkt_trend"]

    fig = market_share_by_year(df, 'market_share_sales', label="Market Share Sales")
    charts["market_share_sales"] = fig
    plt.close(fig)

    fig = market_share_by_year(df, 'market_share_units', label="Market Share Units")
    charts["market_share_units"] = fig
    plt.close(fig)
        
    # Set baseline
    baseline_price = float(exp_price)
    baseline_units = float(df_fcst["units"].sum())

    # only want price and units in the table
    price_grid_df = price_grid_df[["avg_price", "units"]]
    price_grid_df.to_csv("price_grid.csv")

    summary = {
    "baseline_price": baseline_price,
    "baseline_units": baseline_units,
    "elasticity_mid": float(E_rel),
    "sales_breakeven_price": float(sales_breakeven_price),
    "units_breakeven_price": float(units_breakeven_price),
    "units_last_year": float(units_ly),
}
    return {
    "price_grid_df": price_grid_df,
    "summary": summary,
    "charts": charts,
    "meta": meta,
}
