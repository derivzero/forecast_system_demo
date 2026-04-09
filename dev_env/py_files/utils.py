# --- utils.py ---
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, date, timezone
from pandas.tseries.offsets import DateOffset

def missing_values(df, ind):
    """
    Forward-fill only the specified columns that have missing values.
    Returns a Series of any remaining missing counts.
    """
    counts = df[ind].isna().sum()
    missing_vars = counts[counts > 0].index.tolist()

    if missing_vars:
        df.loc[:, missing_vars] = df[missing_vars].ffill()   # or .ffill().bfill() to fill leading NaNs too

    counts_after = df[ind].isna().sum()
    return counts_after[counts_after > 0]

# generic Bennett for a product M = A * B
def decomp_product_bennett(A0, A1, B0, B1):
    dA = A1 - A0
    dB = B1 - B0
    dM_A = 0.5 * (B0 + B1) * dA
    dM_B = 0.5 * (A0 + A1) * dB
    return float(dM_A), float(dM_B)

# Sales = Price * Units
def decomp_sales_bennett(ly_units, ly_sales, ty_units, ty_sales):
    P0 = ly_sales / ly_units if ly_units else 0.0
    P1 = ty_sales / ty_units if ty_units else 0.0
    return decomp_product_bennett(P0, P1, ly_units, ty_units)

# COGS = AvgCost * Units
def decomp_cogs_bennett(ly_units, ly_cogs, ty_units, ty_cogs):
    AC0 = ly_cogs / ly_units if ly_units else 0.0
    AC1 = ty_cogs / ty_units if ty_units else 0.0
    return decomp_product_bennett(AC0, AC1, ly_units, ty_units)

def build_dial_yoy_block(df, end_date, months_back, dials_base, dials_derived, value_type):
    """
    Builds YOY dial values for a given window.
    
    months_back:
        11 for TTM
        5  for 6-month forecast
    """

    ty_end = end_date
    ty_start = ty_end - DateOffset(months=months_back)
    ly_end = ty_end - DateOffset(months=12)
    ly_start = ly_end - DateOffset(months=months_back)

    # --- Aggregate base metrics ---
    ty_sum = {}
    ly_sum = {}

    for col in dials_base:
        ty_sum[f"{col}_ty"] = df.loc[ty_start:ty_end, col].sum()
        ly_sum[f"{col}_ly"] = df.loc[ly_start:ly_end, col].sum()

    ty_df = pd.DataFrame([ty_sum])
    ly_df = pd.DataFrame([ly_sum])
    block_df = pd.concat([ty_df, ly_df], axis=1)

    # --- Derived metrics ---
    block_df["avg_price_trend_ty"] = block_df["sales_mkt_trend_ty"] / block_df["units_mkt_trend_ty"]
    block_df["avg_price_trend_ly"] = block_df["sales_mkt_trend_ly"] / block_df["units_mkt_trend_ly"]

    block_df["avg_price_ty"] = block_df["sales_ty"] / block_df["units_ty"]
    block_df["avg_price_ly"] = block_df["sales_ly"] / block_df["units_ly"]

    block_df["avg_cost_ty"] = block_df["cogs_ty"] / block_df["units_ty"]
    block_df["avg_cost_ly"] = block_df["cogs_ly"] / block_df["units_ly"]

    block_df["upv_ty"] = block_df["units_ty"] / block_df["visits_ty"]
    block_df["upv_ly"] = block_df["units_ly"] / block_df["visits_ly"]

    # --- YOY Calculation ---
    yoy = {}

    for col in (dials_base + dials_derived):
        ty_val = block_df[f"{col}_ty"].iloc[0]
        ly_val = block_df[f"{col}_ly"].iloc[0]
        yoy[col] = ((ty_val / ly_val) - 1) * 100

    yoy_df = pd.DataFrame([yoy])
    yoy_long = yoy_df.melt(var_name="metric", value_name="value")
    yoy_long["value_type"] = value_type

    return yoy_long


def build_ttm(df, vars, end_train):
    """
    Returns long-format DataFrame with:
        period_date
        var
        ly
        ty
        yoy
    For the trailing 12 months ending at end_train.
    """

    end_train = pd.to_datetime(end_train)

    # Need 24 months total
    start = end_train - pd.DateOffset(months=23)
    window = df.loc[start:end_train, vars].copy()

    if len(window) < 24:
        raise ValueError("Need 24 months of data to compute TTM comparison.")

    # Split
    ly = window.iloc[:12].copy()
    ty = window.iloc[12:].copy()

    # Align index (important)
    ly.index = ty.index

    # Compute YOY %
    yoy = (ty / ly - 1.0) * 100.0

    # Build tidy output
    out = []

    for var in vars:
        tmp = pd.DataFrame({
            "period_date": ty.index,
            "var": var,
            "ly": ly[var].values,
            "ty": ty[var].values,
            "yoy": yoy[var].values
        })
        out.append(tmp)

    return pd.concat(out, ignore_index=True)






    




