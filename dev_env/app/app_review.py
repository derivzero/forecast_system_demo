# ----------------------------
# 0. Streamlit MUST be configured first
# ----------------------------
import streamlit as st

st.set_page_config(
    page_title="Grocery Dashboard",
    layout="wide"
)

# ----------------------------
# 1. Tell Python where the project root is
# ----------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------
# 2. Standard library imports
# ----------------------------
import os
import logging
import json
from html import escape
from io import BytesIO

# ----------------------------
# 3. Third-party imports
# ----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# 4. Your project imports (ONLY after sys.path fix)
# ----------------------------
from master import build_artifacts
from py_files.run_configs import RUN_CONFIG as cfg_run
import importlib.util

# Runs the code once and then holds in memory
@st.cache_resource(show_spinner=True)
def load_artifacts():
    return build_artifacts()

df, _, price_grid_df, _, ALL_CHARTS, ALL_META = load_artifacts()

df["period_date"] = pd.to_datetime(df["period_date"])
df = df.set_index("period_date").sort_index()

# Chart Loader function - pulls charts from memory
@st.cache_data(show_spinner=False)
def load_image(path: str) -> bytes:
    """
    Load an image from disk ONCE and cache it in memory.
    After first load, Streamlit serves from RAM.
    """
    return Path(path).read_bytes()

# Turn on Streamlit
print("TOP: app starting", flush=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.info("TOP: logger ready")

# Define slices
start_train = pd.to_datetime(cfg_run.start_train)
end_train   = pd.to_datetime(cfg_run.end_train)
start_fcst = pd.to_datetime(cfg_run.start_fcst)
end_fcst   = pd.to_datetime(cfg_run.end_fcst)

# STREAMLIT Config ------------------------------------------------------------------------------

# This function allows for the creation of the interactive sliders
def scenario_forward_only(
    base,                      # DataFrame with columns listed below
    start_fcst,                # pd.Timestamp
    pct_price=0.0,             # Δ% for price
    pct_avg_cost=0.0,          # Δ% for avg_cost (variable/unit cost)
    pct_fixed=0.0,             # Δ% for fixed cost
    elasticity=0.0,            # units response to price; e.g., -1.2
    *,
    units_col="units",
    price_col="avg_price",
    cost_col="avg_cost",       # <-- use your avg_cost forecast column
    fixed_col="fixed_cost",
    include_fixed_in_gm=False  # GM = Sales-COGS if False; minus fixed if True
):
    out = base.copy()
    mask = out.index >= start_fcst

    mp   = 1 + pct_price/100.0
    mc   = 1 + pct_avg_cost/100.0
    mfix = 1 + pct_fixed/100.0

    # Adjusted price & avg_cost (forecast horizon only)
    out["price_adj"] = out[price_col]
    out.loc[mask, "price_adj"] = out.loc[mask, price_col] * mp

    out["avg_cost_adj"] = out[cost_col]
    out.loc[mask, "avg_cost_adj"] = out.loc[mask, cost_col] * mc

    # Adjusted fixed (only used if include_fixed_in_gm=True)
    out["fixed_adj"] = out[fixed_col] if fixed_col in out.columns else 0.0
    if fixed_col in out.columns:
        out.loc[mask, "fixed_adj"] = out.loc[mask, fixed_col] * mfix

    # Units (optionally react to price)
    out["units_adj"] = out[units_col]
    if float(elasticity) != 0.0:
        out.loc[mask, "units_adj"] = (
            out.loc[mask, units_col] * (1 + float(elasticity) * (pct_price/100.0))
        )
    out["units_adj"] = out["units_adj"].clip(lower=0)

    # Recompute P&L
    out["sales"] = out["price_adj"]    * out["units_adj"]
    out["cogs"]  = out["avg_cost_adj"] * out["units_adj"]

    out["gm"] = out["sales"] - out["cogs"]
    if include_fixed_in_gm and (fixed_col in out.columns):
        out["gm"] = out["gm"] - out["fixed_adj"]

    return out

# Config to clean up and standardize charts --------------------------

plt.rcParams.update({
    # Rendering
    "figure.dpi": 140,                 # crisper figures on screens
    "savefig.dpi": 160,                # crisper saved images
    "figure.constrained_layout.use": True,  # better auto layout than tight_layout in many cases

    # Text sizes
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Look & feel
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.18,                # subtle grid
    "grid.linestyle": "-",             # solid, faint gridlines
    "grid.linewidth": 0.6,

    # Lines & markers
    "lines.linewidth": 2,
    "lines.markersize": 5,

    # Legend box
    "legend.frameon": False,
})

# Handy money formatter you can reuse:
currency = FuncFormatter(lambda x, _: f'${x:,.0f}')

# ---- Sidebar Nav: two grouped radios, mutually exclusive ----

# init defaults
st.session_state.setdefault("active_tab", "Home")
st.session_state.setdefault("market_nav", None)
st.session_state.setdefault("retailer_ttm_nav", None)
st.session_state.setdefault("retailer_demand_fcst_nav", None)
st.session_state.setdefault("retailer_cost_fcst_nav", None)
st.session_state.setdefault("retailer_margin_fcst_nav", None)
st.session_state.setdefault("optimization_nav", None)

def choose_market():
    st.session_state["retailer_ttm_nav"] = None
    st.session_state["retailer_demand_fcst_nav"] = None
    st.session_state["retailer_cost_fcst_nav"] = None
    st.session_state["retailer_margin_fcst_nav"] = None
    st.session_state["optimization_nav"] = None
    st.session_state["active_tab"] = st.session_state["market_nav"]

def choose_retailer_ttm():
    st.session_state["market_nav"] = None
    st.session_state["retailer_demand_fcst_nav"] = None
    st.session_state["retailer_cost_fcst_nav"] = None
    st.session_state["retailer_margin_fcst_nav"] = None
    st.session_state["optimization_nav"] = None
    st.session_state["active_tab"] = st.session_state["retailer_ttm_nav"]

def choose_retailer_demand_fcst():
    st.session_state["market_nav"] = None
    st.session_state["retailer_ttm_nav"] = None
    st.session_state["optimization_nav"] = None
    st.session_state["retailer_cost_fcst_nav"] = None
    st.session_state["retailer_margin_fcst_nav"] = None
    st.session_state["active_tab"] = st.session_state["retailer_demand_fcst_nav"]

def choose_retailer_cost_fcst():
    st.session_state["market_nav"] = None
    st.session_state["retailer_ttm_nav"] = None
    st.session_state["optimization_nav"] = None
    st.session_state["retailer_demand_fcst_nav"] = None
    st.session_state["retailer_margin_fcst_nav"] = None
    st.session_state["active_tab"] = st.session_state["retailer_cost_fcst_nav"]

def choose_retailer_margin_fcst():
    st.session_state["market_nav"] = None
    st.session_state["retailer_ttm_nav"] = None
    st.session_state["optimization_nav"] = None
    st.session_state["retailer_demand_fcst_nav"] = None
    st.session_state["retailer_cost_fcst_nav"] = None
    st.session_state["active_tab"] = st.session_state["retailer_margin_fcst_nav"]

def choose_optimization():
    st.session_state["market_nav"] = None
    st.session_state["retailer_ttm_nav"] = None
    st.session_state["retailer_demand_fcst_nav"] = None
    st.session_state["retailer_cost_fcst_nav"] = None
    st.session_state["retailer_margin_fcst_nav"] = None
    st.session_state["active_tab"] = st.session_state["optimization_nav"]

st.sidebar.subheader("Market")
st.sidebar.radio(
    label="Market navigation",             # non-empty!
    options=["Home", "Market Charts TTM", "CPI FAH Forecast", "US Units Forecast", "US Sales Forecast" ],
    index=None,
    key="market_nav",
    label_visibility="collapsed",          # hides it visually, keeps accessibility
    on_change=choose_market,
)

st.sidebar.subheader("Retailer History (TTM)")
st.sidebar.radio(
    label="Retailer Navigation TTM",           # non-empty!
    options=["KPI Snapshot TTM", "KPI Trends TTM", "KPI Tree TTM"],
    index=None,
    key="retailer_ttm_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_ttm,
)

st.sidebar.subheader("Retailer Demand Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Demand Forecast",           # non-empty!
    options=["Unit Forecast", "Sales Forecast", "Visit Forecast", "UPV Forecast"],
    index=None,
    key="retailer_demand_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_demand_fcst,
)

st.sidebar.subheader("Retailer Cost Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Cost Forecast",           # non-empty!
    options=["Average Cost Forecast", "COGS Forecast", "Total Cost Forecast"],
    index=None,
    key="retailer_cost_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_cost_fcst,
)

st.sidebar.subheader("Retailer Margin Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Margin Forecast",           # non-empty!
    options=["Gross Margin Forecast", "Net Income Forecast", "KPI Tree Forecast"],
    index=None,
    key="retailer_margin_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_margin_fcst,
)

st.sidebar.subheader("Optimization")
st.sidebar.radio(
    label="Optimization",           # non-empty!
    options=["Gross Margin, Net Income, and Market Share Trends", "Financial Simulation"],
    index=None,
    key="optimization_nav",
    label_visibility="collapsed",
    on_change=choose_optimization,
)

# Use this everywhere below
tab = st.session_state["active_tab"]

# -----------------------------------------------------------------------------------------------
# HOME (landing page)
# -----------------------------------------------------------------------------------------------
if tab == "Home":

    st.title("U.S. Grocery Market Navigator")
    st.header("Leverage forward looking ML algorithms to optimize financial performance")
    st.divider()
    IMG_DIR = Path(__file__).resolve().parents[1] / "app" / "tab0"
    logo_path = IMG_DIR / "dz_logo.jpg"
    left, r1c1, right = st.columns([0.2, 0.4, 0.45])
    r1c1.image(load_image(str(logo_path)), use_container_width=True)

# -----------------------------------------------------------------------------------------------
# TAB 1 - Market Trends
# -----------------------------------------------------------------------------------------------
elif tab == "Market Charts TTM":
    st.header("U.S. Grocery Market Data: Trailing 12 Month (TTM) Data")
    
    start = end_train - pd.DateOffset(months=23)
    end = end_train
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['units_mkt_trend', 'sales_mkt_trend']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy = (ty_sum / ly_sum - 1) * 100

    ty_mean = df_yoy.loc[ty_mask, 'cpi_fah'].mean()
    ly_mean = df_yoy.loc[ly_mask, 'cpi_fah'].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # append into yoy series
    yoy['cpi_fah'] = yoy_mean
    
    # add yoy suffix
    yoy_dial = yoy.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig

    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=18):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        header_center("U.S. CPI FAH TTM YOY") 
        st.plotly_chart(gauge(yoy_dial['cpi_fah_yoy']), use_container_width=True)
    with r1c2:
        header_center("U.S. Grocery Market Units TTM YOY")   
        st.plotly_chart(gauge(yoy_dial['units_mkt_trend_yoy']), use_container_width=True)
    with r1c3:
        header_center("U.S. Grocery Market Sales TTM YOY")   
        st.plotly_chart(gauge(yoy_dial['sales_mkt_trend_yoy']), use_container_width=True)

    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        fig = ALL_CHARTS["cpi_fah_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r2c2:
        fig = ALL_CHARTS["units_mkt_trend_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r2c3:
        fig = ALL_CHARTS["sales_mkt_trend_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 2 - CPI FAH Forecast
# -------------------------------------------------------------------------------------
elif tab == "CPI FAH Forecast":
    import numpy as np
    import pandas as pd
    
    st.header("CPI Food-at-home")
        
    # ----- Dial: 6-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_6 = df.loc[ty_idx, 'cpi_fah'].mean()
    ly_6 = df['cpi_fah'].shift(12).loc[ty_idx].mean()
    dial6 = (ty_6 / ly_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['cpi_fah', 'oil_prices_lag7', 'ppi_farm_products_lag4', 'ppi_food_mfg_lag3', 'ppi_grocery']
    ty_mean = df_yoy.loc[ty_mask, vars].mean()
    ly_mean = df_yoy.loc[ly_mask, vars].mean()
    yoy12 = (ty_mean / ly_mean - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy12.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=10):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2  = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['cpi_fah_yoy']))
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6))
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        header_center("U.S. Oil Price L7 TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['oil_prices_lag7_yoy']), use_container_width=True)
    with r2c2:
        header_center("U.S. PPI Farm Products L4 TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['ppi_farm_products_lag4_yoy']), use_container_width=True)
    with r2c3:
        header_center("U.S. PPI Food Manufacture L3 TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['ppi_food_mfg_lag3_yoy']), use_container_width=True)
    with r2c4:
        header_center("U.S. PPI Grocery GM TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['ppi_grocery_yoy']), use_container_width=True)
    
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        fig = ALL_CHARTS["oil_prices_lag7_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r3c2:
        fig = ALL_CHARTS["ppi_farm_products_lag4_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r3c3:
        fig = ALL_CHARTS["ppi_food_mfg_lag3_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c4:
        fig = ALL_CHARTS["ppi_grocery_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Driver Importance")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["cpi_fah_betas"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["cpi_fah_betas_normalized"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
        
        
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["cpi_fah_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["cpi_fah_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)


    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r6c1, r6c2 = st.columns(2)
    with r6c1:
        fig = ALL_CHARTS["cpi_fah_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r6c2:
        fig = ALL_CHARTS["cpi_fah_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    
    st.divider() # adds a line for white space
    st.subheader("Holdout Analysis and Model Performance")
    
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        fig = ALL_CHARTS["cpi_fah_forecast_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r7c2:
        fig = ALL_CHARTS["cpi_fah_rolling_6m_mape_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# ----------------------------------------------------------------------------------------
# TAB 3 - US Units
# ----------------------------------------------------------------------------------------
elif tab == "US Units Forecast":
    st.header("US Units")
    
    # ----- Dial: 6-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_6 = df.loc[ty_idx, 'units_mkt_trend'].sum()
    ly_6 = df['units_mkt_trend'].shift(12).loc[ty_idx].sum()
    dial6 = (ty_6 / ly_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['units_mkt_trend', 'rdi', 'home_price']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    ty_mean = df_yoy.loc[ty_mask, 'cpi_fah'].mean()
    ly_mean = df_yoy.loc[ly_mask, 'cpi_fah'].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # append into yoy series
    yoy_sum['cpi_fah'] = yoy_mean

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=10):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['units_mkt_trend_yoy']))
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6))
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
        
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        header_center("U.S. CPI FAH TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['cpi_fah_yoy']), use_container_width=True)
    with r2c2:
        header_center("U.S. RDI TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['rdi_yoy']), use_container_width=True)
    with r2c3:
        header_center("U.S. HOME PRICE TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['home_price_yoy']), use_container_width=True)
            
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        fig = ALL_CHARTS["cpi_fah_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r3c2:
        fig = ALL_CHARTS["rdi_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r3c3:
        fig = ALL_CHARTS["home_price_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
       
    st.divider() # adds a line for white space
    st.subheader("Driver Importance")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["units_mkt_trend_betas"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r4c2:
        fig = ALL_CHARTS["units_mkt_trend_betas_normalized"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

        
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["units_mkt_trend_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r5c2:
        fig = ALL_CHARTS["units_mkt_trend_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    
    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r6c1, r6c2 = st.columns(2)
    with r6c1:
        fig = ALL_CHARTS["units_mkt_trend_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r6c2:
        fig = ALL_CHARTS["units_mkt_trend_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
 
    
    st.divider() # adds a line for white space
    st.subheader("Holdout Analysis and Model Performance")
    
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        fig = ALL_CHARTS["units_mkt_trend_forecast_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r7c2:
        fig = ALL_CHARTS["units_mkt_trend_rolling_6m_mape_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
 

# -------------------------------------------------------------------------------------
# TAB 4 - US Sales Forecast
# -------------------------------------------------------------------------------------
elif tab == "US Sales Forecast":
    st.header("US Sales")
    
    # ----- Dial: 6-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_6 = df.loc[ty_idx, 'sales_mkt_trend'].sum()
    ly_6 = df['sales_mkt_trend'].shift(12).loc[ty_idx].sum()
    dial6 = (ty_6 / ly_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['sales_mkt_trend', 'units_mkt_trend']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    ty_mean = df_yoy.loc[ty_mask, 'cpi_fah'].mean()
    ly_mean = df_yoy.loc[ly_mask, 'cpi_fah'].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # append into yoy series
    yoy_sum['cpi_fah'] = yoy_mean

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=10):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2) # relative spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['sales_mkt_trend_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
        
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("U.S. CPI FAH TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['cpi_fah_yoy']), use_container_width=True)
    with r2c2:
        header_center("U.S. Units TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['units_mkt_trend_yoy']), use_container_width=True)
       
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["cpi_fah_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r3c2:
        fig = ALL_CHARTS["units_mkt_trend_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["sales_mkt_trend_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r4c2:
        fig = ALL_CHARTS["sales_mkt_trend_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["sales_mkt_trend_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    with r5c2:
        fig = ALL_CHARTS["sales_mkt_trend_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# ----------------------------------------------------------------------------------------
# TAB 5 - KPI Snapshot
# ----------------------------------------------------------------------------------------
elif tab == "KPI Snapshot TTM":
    
    start = end_train - pd.DateOffset(months=23)
    end = end_train
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    vars = ['units', 'sales', 'cogs', 'gm']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy = (ty_sum / ly_sum - 1) * 100

    # create avg_price and avg_cost
    avg_price_yoy = ((ty_sum['sales'] / ty_sum['units']) / (ly_sum['sales'] / ly_sum['units']) - 1) * 100
    avg_cost_yoy = ((ty_sum['cogs'] / ty_sum['units']) / (ly_sum['cogs'] / ly_sum['units']) - 1) * 100

    # append into yoy series
    yoy['avg_price'] = avg_price_yoy
    yoy['avg_cost'] = avg_cost_yoy

    # add yoy suffix
    yoy_series = yoy.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(title, value, vmin=-20, vmax=20, *, height=200, number_size=34, title_size=18, top_margin=60):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={'text': title, 'font': {'size': title_size}},
            number={'suffix': '%', 'valueformat': '.1f', 'font': {'size': number_size}},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=top_margin, b=10))
        return fig

    # row 1
    c1, = st.columns(1)
    c1.plotly_chart(gauge("Units TTM YOY", yoy_series['units_yoy'], height=260), use_container_width=True)

    st.divider() # adds a line for white space

    # row 2
    c2, c3 = st.columns(2)
    c2.plotly_chart(gauge("Avg Price TTM YOY", yoy_series['avg_price_yoy']), use_container_width=True)
    c3.plotly_chart(gauge("Avg Cost TTM YOY", yoy_series['avg_cost_yoy']), use_container_width=True)

    st.divider() # adds a line for white space

    # Row 3
    c4, c5 = st.columns(2)
    c4.plotly_chart(gauge("Sales TTM YOY", yoy_series['sales_yoy']), use_container_width=True)
    c5.plotly_chart(gauge("COGS TTM YOY", yoy_series['cogs_yoy']), use_container_width=True)

    st.divider() # adds a line for white space
    
    # Row 4
    c6, = st.columns(1)
    c6.plotly_chart(gauge("Gross Margin Dollars TTM YOY", yoy_series['gm_yoy'], height=260), use_container_width=True)
        
# ----------------------------------------------------------------------------------------
# TAB 6 - KPI Trends
# ----------------------------------------------------------------------------------------
elif tab == "KPI Trends TTM":
    st.header("KPI Trends: Current Year (TTM) vs Previous Year")

    start = end_train - pd.DateOffset(months=23)
    end = end_train
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    vars = ['units', 'avg_price', 'avg_cost', 'sales', 'cogs', 'gm']
    ly_trend = df_yoy.loc[ly_mask, vars]
    ty_trend = df_yoy.loc[ty_mask, vars]

    # align months for overlay
    ly_aligned = ly_trend.copy()
    ly_aligned.index = ly_aligned.index + pd.DateOffset(years=1)
    ly_aligned = ly_aligned.reindex(ty_trend.index)

    # 3) MONTHLY YoY% = (TY / LY - 1) * 100, safe divide
    yoy_pct = (ty_trend / ly_aligned).replace([np.inf, -np.inf], np.nan) - 1.0
    yoy_pct = yoy_pct * 100.0

    def plot_metric(name, title, yaxis_title):
        df_plot = pd.DataFrame({
            "TY": ty_trend[name],
            "LY": ly_aligned[name],
            "YoY%": yoy_pct[name]
        })

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["TY"], name="Current Year (TTM)", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["LY"], name="Previous Year", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Bar(x=df_plot.index, y=df_plot["YoY%"], name="YoY %"), secondary_y=True)

        fig.update_layout(
            title=title,
            margin=dict(l=10, r=10, t=40, b=10),
            height=340,
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text=yaxis_title, secondary_y=False)
        fig.update_yaxes(title_text="YoY (%)", secondary_y=True)
        
        # Make YoY bars shorter by expanding the secondary y-axis range
        ymax = float(np.nanmax(np.abs(df_plot["YoY%"])))
        cap  = max(20.0, round((ymax * 2) / 5) * 5)  # at least ±20, rounded to nearest 5
        fig.update_yaxes(range=[-cap, cap], secondary_y=True)

        # Nice touch: slightly transparent bars so lines stay on top
        fig.update_traces(opacity=0.3, selector=dict(type="bar"))
        
        return fig

    nice_names = {
        "units": "Units",
        "avg_price": "Avg Price",
        "avg_cost": "Avg Cost",
        "sales": "Sales",
        "cogs": "COGS",
        "gm": "GM($)",
    }
    y_axis_label = {
        "units": "Units",
        "avg_price": "Dollars",
        "avg_cost": "Dollars",
        "sales": "Dollars",
        "cogs": "Dollars",
        "gm": "Dollars",
    }

    metrics_order = ["units", "avg_price", "avg_cost", "sales", "cogs", "gm"]

    # row 1
    c1, = st.columns(1)
    c1.plotly_chart(plot_metric("units", nice_names["units"], y_axis_label["units"]), use_container_width=True)
    
    st.divider() # adds a line for white space

    c2, c3 = st.columns(2)
    c2.plotly_chart(plot_metric("avg_price", nice_names["avg_price"], y_axis_label["avg_price"]), use_container_width=True)
    c3.plotly_chart(plot_metric("avg_cost", nice_names["avg_cost"], y_axis_label["avg_cost"]), use_container_width=True)

    st.divider() # adds a line for white space

    c4, c5 = st.columns(2)
    c4.plotly_chart(plot_metric("sales", nice_names["sales"], y_axis_label["sales"]), use_container_width=True)
    c5.plotly_chart(plot_metric("cogs", nice_names["cogs"], y_axis_label["cogs"]), use_container_width=True)

    st.divider() # adds a line for white space
    
    c6, = st.columns(1)
    c6.plotly_chart(plot_metric("gm", nice_names["gm"], y_axis_label["gm"]), use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 7 - KPI Tree TTM
# -------------------------------------------------------------------------------------
elif tab == "KPI Tree TTM":
    import pandas as pd
    from pathlib import Path
    from py_files.utils import decomp_sales_bennett, decomp_cogs_bennett

    st.header("KPI Tree: Trailing 12 Months (TTM)")

    # --- Build yoy_series locally for this tab (12m TY vs 12m LY) ---
    start = end_train - pd.DateOffset(months=23)  # 24 months inclusive
    end   = end_train
    df24  = df.loc[start:end]

    ly_idx = df24.head(12).index   # LY
    ty_idx = df24.tail(12).index   # TY

    cols   = ['units', 'sales', 'cogs', 'gm']
    ty_sum = df24.loc[ty_idx, cols].sum()
    ly_sum = df24.loc[ly_idx, cols].sum()

    dS_total = float(ty_sum['sales'] - ly_sum['sales'])
    dS_price, dS_units = decomp_sales_bennett(ly_sum['units'], ly_sum['sales'], ty_sum['units'], ty_sum['sales'])
    
    dC_total = float(ty_sum['cogs'] - ly_sum['cogs'])
    dC_cost, dC_units = decomp_cogs_bennett(ly_sum['units'], ly_sum['cogs'], ty_sum['units'], ty_sum['cogs'])

    # YOY % for core metrics
    yoy = (ty_sum / ly_sum - 1) * 100
    
    # Derived averages via 12m sums (weighted by units)
    yoy['avg_price'] = ((ty_sum['sales']/ty_sum['units']) / (ly_sum['sales']/ly_sum['units']) - 1) * 100
    yoy['avg_cost']  = ((ty_sum['cogs'] /ty_sum['units'])  / (ly_sum['cogs'] /ly_sum['units'])  - 1) * 100
    yoy_series = yoy.add_suffix('_yoy')

    # --- Dollar deltas for Sales/COGS/GM (12m TY vs 12m LY) ---
    S0, C0, G0 = ly_sum['sales'], ly_sum['cogs'], ly_sum['gm']
    S1, C1, G1 = ty_sum['sales'], ty_sum['cogs'], ty_sum['gm']
    dS, dC, dG = S1 - S0, C1 - C0, G1 - G0

    # money formatters
    def fmt_money(x):
        ax = abs(x)
        if ax >= 1e9:  return f"${x/1e9:.1f}B"
        if ax >= 1e6:  return f"${x/1e6:.1f}M"
        if ax >= 1e3:  return f"${x/1e3:.0f}K"
        return f"${x:.0f}"
    
    def signed(x): return ("+" if x >= 0 else "−") + fmt_money(abs(x))

    y = yoy_series  # shorthand

    # --- Graphviz DOT (YOY on all; Δ$ only on Sales/COGS/GM) ---
    dot = f"""
    digraph {{
      rankdir=TB;
      nodesep=0.75; ranksep=0.35;
      node [shape=box, style="rounded,filled", color="#cccccc", fillcolor="white", fontsize=12];

      GM  [label="GM$ YoY\\n{y['gm_yoy']:.1f}%\\nΔ {signed(dG)}"];
      S   [label="Sales YoY\\n{y['sales_yoy']:.1f}%\\nΔ {signed(dS)}"];
      C   [label="COGS YoY\\n{y['cogs_yoy']:.1f}%\\nΔ {signed(dC)}"];
      
      U1  [label="Units YoY\\n{y['units_yoy']:.1f}%\\nΔ {signed(dS_units)}"];
      P   [label="Avg Price YoY\\n{y['avg_price_yoy']:.1f}%\\nΔ {signed(dS_price)}"];

      U2  [label="Units YoY\\n{y['units_yoy']:.1f}%\\nΔ {signed(dC_units)}"];
      AC  [label="Avg Cost YoY\\n{y['avg_cost_yoy']:.1f}%\\nΔ {signed(dC_cost)}"];

      U1 -> S;  P  -> S;  S -> GM;
      U2 -> C;  AC -> C;  C -> GM;
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 8 - Unit Forecast
# -------------------------------------------------------------------------------------
elif tab == "Unit Forecast":
    st.header("Retailer Units")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'units'].sum()
    ly_sales_6 = df['units'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['sales', 'units', 'rdi']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    ty_mean = df_yoy.loc[ty_mask, 'home_price'].mean()
    ly_mean = df_yoy.loc[ly_mask, 'home_price'].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # append into yoy series
    yoy_sum['home_price'] = yoy_mean

    yoy_sum['avg_price'] = ((ty_sum['sales']/ty_sum['units']) / (ly_sum['sales']/ly_sum['units']) - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')
    
    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['units_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        header_center("Average Price TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['avg_price_yoy']), use_container_width=True)
    with r2c2:
        header_center("U.S. RDI TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['rdi_yoy']), use_container_width=True)
    with r2c3:
        header_center("U.S. HOME PRICE TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['home_price_yoy']), use_container_width=True)
     
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        fig = ALL_CHARTS["avg_price_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["rdi_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c3:
        fig = ALL_CHARTS["home_price_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Driver Importance")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["units_betas"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["units_betas_normalized"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["units_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["units_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r6c1, r6c2 = st.columns(2)
    with r6c1:
        fig = ALL_CHARTS["units_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r6c2:
        fig = ALL_CHARTS["units_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Holdout Analysis and Model Performance")
    
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        fig = ALL_CHARTS["units_forecast_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r7c2:
        fig = ALL_CHARTS["units_rolling_6m_mape_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 9 - Sales Forecast
# -------------------------------------------------------------------------------------
elif tab == "Sales Forecast":
    st.header("Retailer Sales")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'sales'].sum()
    ly_sales_6 = df['sales'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['sales', 'units']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    yoy_sum['avg_price'] = ((ty_sum['sales']/ty_sum['units']) / (ly_sum['sales']/ly_sum['units']) - 1) * 100
    
    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['sales_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Average Price TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['avg_price_yoy']), use_container_width=True)
    with r2c2:
        header_center("Units TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['units_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["avg_price_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["units_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["sales_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["sales_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["sales_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["sales_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
# -------------------------------------------------------------------------------------
# TAB 10 - Visits Forecast
# -------------------------------------------------------------------------------------
elif tab == "Visit Forecast":
    st.header("Retailer Visits")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'visits'].sum()
    ly_sales_6 = df['visits'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['visits', 'sales', 'units']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    yoy_sum['avg_price'] = ((ty_sum['sales']/ty_sum['units']) / (ly_sum['sales']/ly_sum['units']) - 1) * 100
    
    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['visits_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Average Price TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['avg_price_yoy']), use_container_width=True)
    with r2c2:
        header_center("Units TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['units_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["avg_price_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["units_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Driver Importance")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["visits_betas"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["visits_betas_normalized"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["visits_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["visits_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r6c1, r6c2 = st.columns(2)
    with r6c1:
        fig = ALL_CHARTS["visits_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r6c2:
        fig = ALL_CHARTS["visits_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Holdout Analysis and Model Performance")
    
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        fig = ALL_CHARTS["visits_forecast_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r7c2:
        fig = ALL_CHARTS["visits_rolling_6m_mape_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 11 - UPV Forecast
# -------------------------------------------------------------------------------------
elif tab == "UPV Forecast":
    st.header("Retailer Units per Visit")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'upv'].sum()
    ly_sales_6 = df['upv'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['upv', 'visits', 'units']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['upv_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Visits TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['visits_yoy']), use_container_width=True)
    with r2c2:
        header_center("Units TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['units_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["visits_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["units_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["upv_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["upv_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["upv_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["upv_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 12 - Avg Cost Forecast
# -------------------------------------------------------------------------------------
elif tab == "Average Cost Forecast":
    st.header("Retailer Average Cost")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'avg_cost'].mean()
    ly_sales_6 = df['avg_cost'].shift(12).loc[ty_idx].mean()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    var = ['avg_cost','ppi_food_mfg_lag3']
    ty_mean = df_yoy.loc[ty_mask, var].mean()
    ly_mean = df_yoy.loc[ly_mask, var].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_mean.add_suffix('_yoy')
    
    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['avg_cost_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, = st.columns(1)
    with r2c1:
        header_center("Average Cost TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['ppi_food_mfg_lag3_yoy']), use_container_width=True)
         
    left, r3c1, right = st.columns([0.5, 1, 0.5])
    with r3c1:
        fig = ALL_CHARTS["ppi_food_mfg_lag3_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Driver Importance")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["avg_cost_betas"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["avg_cost_betas_normalized"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
        
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["avg_cost_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["avg_cost_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r6c1, r6c2 = st.columns(2)
    with r6c1:
        fig = ALL_CHARTS["avg_cost_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r6c2:
        fig = ALL_CHARTS["avg_cost_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Holdout Analysis and Model Performance")
    
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        fig = ALL_CHARTS["avg_cost_forecast_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r7c2:
        fig = ALL_CHARTS["avg_cost_rolling_6m_mape_holdout"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 13 - COGS Forecast
# -------------------------------------------------------------------------------------
elif tab == "COGS Forecast":
    st.header("Retailer Cost of Goods Sold")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:cfg_run.end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'cogs'].sum()
    ly_sales_6 = df['cogs'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['cogs', 'units']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    var = 'avg_cost'
    ty_mean = df_yoy.loc[ty_mask, var].mean()
    ly_mean = df_yoy.loc[ly_mask, var].mean()
    yoy_mean = (ty_mean / ly_mean - 1) * 100

    # append into yoy series
    yoy_sum['avg_cost'] = yoy_mean

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['cogs_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Average Cost TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['avg_cost_yoy']), use_container_width=True)
    with r2c2:
        header_center("Units TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['units_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["avg_cost_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["units_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["cogs_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["cogs_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["cogs_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["cogs_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)


# -------------------------------------------------------------------------------------
# TAB 14 - Total Cost Forecast
# -------------------------------------------------------------------------------------
elif tab == "Total Cost Forecast":
    st.header("Retailer Total Cost")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:cfg_run.end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'total_cost'].sum()
    ly_sales_6 = df['total_cost'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['total_cost', 'cogs', 'fixed_cost']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['total_cost_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("COGS TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['cogs_yoy']), use_container_width=True)
    with r2c2:
        header_center("Fixed Cost TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['fixed_cost_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["fixed_cost_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["cogs_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["total_cost_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["total_cost_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["total_cost_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["total_cost_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)


# -------------------------------------------------------------------------------------
# TAB 15 - GM Forecast
# -------------------------------------------------------------------------------------
elif tab == "Gross Margin Forecast":
    st.header("Retailer Gross Margin")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'gm'].sum()
    ly_sales_6 = df['gm'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['gm', 'sales', 'cogs']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['gm_yoy']), use_container_width=True)
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy), use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Sales TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['sales_yoy']), use_container_width=True)
    with r2c2:
        header_center("COGS TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['cogs_yoy']), use_container_width=True)
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["sales_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["cogs_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["gm_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["gm_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["gm_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["gm_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 16 - NI Forecast
# -------------------------------------------------------------------------------------
elif tab == "Net Income Forecast":
    st.header("Retailer Net Income")
    
    # ----- Dial: 6-month Sales YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    ty_idx = df.loc[start_fcst:end_fcst].index
    ty_sales_6 = df.loc[ty_idx, 'net_income'].sum()
    ly_sales_6 = df['net_income'].shift(12).loc[ty_idx].sum()
    dial6_yoy = (ty_sales_6 / ly_sales_6 - 1.0) * 100.0

    # ----- Dial: 12-month Units YoY forecast (TY = forecast window; LY = same months last year)
    # Assumes start_date_forecast and end_date_forecast already defined earlier in your app
    
    end = end_train
    start = end_train - pd.DateOffset(months=23)
    
    df_yoy = df.loc[start:end].copy()
    ly_mask = df_yoy.head(12).index
    ty_mask = df_yoy.tail(12).index
    
    vars = ['net_income', 'sales', 'total_cost']
    ty_sum = df_yoy.loc[ty_mask, vars].sum()
    ly_sum = df_yoy.loc[ly_mask, vars].sum()
    yoy_sum = (ty_sum / ly_sum - 1) * 100

    # add yoy suffix
    dial12_yoy = yoy_sum.add_suffix('_yoy')

    # tiny helper to draw one dial
    def gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={'suffix': '%', 'valueformat': '.1f'},
            gauge={'axis': {'range': [vmin, vmax]}}
        ))
        fig.update_layout(height=height, margin=dict(l=20, r=20, t=top_margin, b=10))
        return fig
    
    # function to center titles
    def header_center(txt, level=3, bottom_px=6, font_px=16):
        st.markdown(
            f"<h{level} style='text-align:center; font-size:{font_px}px; margin:0 0 {bottom_px}px 0'>{txt}</h{level}>",
        unsafe_allow_html=True,
    )

    # --- Row 1: three charts -------------------------------------------------
    r1c1, r1c2 = st.columns(2)  # spacers
    with r1c1:
        header_center("Actuals: TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['net_income_yoy']))
    with r1c2:
        header_center("Forecast: 6-month YOY")
        st.plotly_chart(gauge(dial6_yoy))
    
    st.divider() # adds a line for white space
    st.subheader("Primary Forecast Inputs TTM")
    
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        header_center("Sales TTM YOY") 
        st.plotly_chart(gauge(dial12_yoy['sales_yoy']), use_container_width=True)
    with r2c2:
        header_center("Total Cost TTM YOY")   
        st.plotly_chart(gauge(dial12_yoy['total_cost_yoy']), use_container_width=True)
        
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = ALL_CHARTS["sales_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r3c2:
        fig = ALL_CHARTS["total_cost_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    
    st.divider() # adds a line for white space
    st.subheader("Forecast: 6-month Outlook")
    
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        fig = ALL_CHARTS["net_income_level_forecast"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r4c2:
        fig = ALL_CHARTS["net_income_level_forecast_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
        
    st.divider() # adds a line for white space
    st.subheader("Forecast Distributions: 6-month Outlook")

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        fig = ALL_CHARTS["net_income_forecast_distribution"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r5c2:
        fig = ALL_CHARTS["net_income_forecast_distribution_yoy"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 17 - KPI Tree Forecast
# -------------------------------------------------------------------------------------

elif tab == "KPI Tree Forecast":
    import pandas as pd
    from pathlib import Path
    from py_files.utils import decomp_sales_bennett, decomp_cogs_bennett

    st.header("KPI Tree Forecast: 6-month Outlook")

    # --- Build yoy_series locally for this tab (6m TY vs 6m LY) ---
    cols   = ['units', 'sales', 'cogs', 'gm']
    # Use the same monthly-collapsed df_m from above
    start = pd.to_datetime(start_fcst)
    end   = pd.to_datetime(end_fcst)

    ty_win = df.loc[start : end]
    ly_win = df.loc[start - pd.DateOffset(years=1) : end - pd.DateOffset(years=1)] # need to subtract one day to take us back to last day of previous month

    ty_sum = ty_win[cols].sum()
    ly_sum = ly_win[cols].sum()

    dS_total = float(ty_sum['sales'] - ly_sum['sales'])
    dS_price, dS_units = decomp_sales_bennett(ly_sum['units'], ly_sum['sales'], ty_sum['units'], ty_sum['sales'])
    
    dC_total = float(ty_sum['cogs'] - ly_sum['cogs'])
    dC_cost, dC_units = decomp_cogs_bennett(ly_sum['units'], ly_sum['cogs'], ty_sum['units'], ty_sum['cogs'])

    # YOY % for core metrics
    yoy = (ty_sum / ly_sum - 1) * 100
    
    # Derived averages via 6m sums (weighted by units)
    yoy['avg_price'] = ((ty_sum['sales']/ty_sum['units']) / (ly_sum['sales']/ly_sum['units']) - 1) * 100
    yoy['avg_cost']  = ((ty_sum['cogs'] /ty_sum['units'])  / (ly_sum['cogs'] /ly_sum['units'])  - 1) * 100
    yoy_series = yoy.add_suffix('_yoy')

    # --- Dollar deltas for Sales/COGS/GM (6m TY vs 6m LY) ---
    S0, C0, G0 = ly_sum['sales'], ly_sum['cogs'], ly_sum['gm']
    S1, C1, G1 = ty_sum['sales'], ty_sum['cogs'], ty_sum['gm']
    dS, dC, dG = S1 - S0, C1 - C0, G1 - G0

    # money formatters
    def fmt_money(x):
        ax = abs(x)
        if ax >= 1e9:  return f"${x/1e9:.1f}B"
        if ax >= 1e6:  return f"${x/1e6:.1f}M"
        if ax >= 1e3:  return f"${x/1e3:.1f}K"
        return f"${x:.0f}"
    
    def signed(x): return ("+" if x >= 0 else "−") + fmt_money(abs(x))

    y = yoy_series  # shorthand

    # --- Graphviz DOT (YOY on all; Δ$ only on Sales/COGS/GM) ---
    dot = f"""
    digraph {{
      rankdir=TB;
      nodesep=0.75; ranksep=0.35;
      node [shape=box, style="rounded,filled", color="#cccccc", fillcolor="white", fontsize=12];

      GM  [label="GM$ YoY\\n{y['gm_yoy']:.1f}%\\nΔ {signed(dG)}"];
      S   [label="Sales YoY\\n{y['sales_yoy']:.1f}%\\nΔ {signed(dS)}"];
      C   [label="COGS YoY\\n{y['cogs_yoy']:.1f}%\\nΔ {signed(dC)}"];
      
      U1  [label="Units YoY\\n{y['units_yoy']:.1f}%\\nΔ {signed(dS_units)}"];
      P   [label="Avg Price YoY\\n{y['avg_price_yoy']:.1f}%\\nΔ {signed(dS_price)}"];

      U2  [label="Units YoY\\n{y['units_yoy']:.1f}%\\nΔ {signed(dC_units)}"];
      AC  [label="Avg Cost YoY\\n{y['avg_cost_yoy']:.1f}%\\nΔ {signed(dC_cost)}"];

      U1 -> S;  P  -> S;  S -> GM;
      U2 -> C;  AC -> C;  C -> GM;
    }}
    """
    st.graphviz_chart(dot, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 18 - Long Run charts
# -------------------------------------------------------------------------------------

elif tab == "Gross Margin, Net Income, and Market Share Trends":
    import pandas as pd
    import json
    import plotly.graph_objects as go
    from pathlib import Path

    st.header("Gross Margin, Net Income, and Market Share Trends")
    st.divider() # adds a line for white space

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = ALL_CHARTS["avg_price_and_gm"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r1c2:
        fig = ALL_CHARTS["avg_price_and_net_income"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

    st.divider() # adds a line for white space
    st.subheader("Market Share")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = ALL_CHARTS["market_share_sales"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)
    with r2c2:
        fig = ALL_CHARTS["market_share_units"]
        st.pyplot(fig=fig, clear_figure=False, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 19 - Financial Simulation
# -------------------------------------------------------------------------------------

elif tab == "Financial Simulation":
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    from html import escape

    st.header("Financial Performance Optimization: 6-month Outlook")

    # --- Elasticity dial ---
    E_rel = ALL_META["units_E_rel"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=E_rel,
        title={"text": "Price Elasticity"},
        gauge={"axis": {"range": [-3, 0]}}
    ))
    left, r1c1, right = st.columns([0.4, 1, 0.4])
    with r1c1:
        st.plotly_chart(fig, use_container_width=True)

    
    # --- price range -----
    st.divider()
    st.subheader("Min and Max Price Range: 6-month Outlook")    
    
    fig = ALL_CHARTS["price_range"]
    left, r3c1, right = st.columns([0.1, 1, 0.1])
    with r3c1:
        st.pyplot(fig=fig, use_container_width=True)

    st.divider()
    st.header("Scenario Planning")

    # --- Forecast and LY windows ---
    ty = df.loc[cfg_run.start_fcst : cfg_run.end_fcst]
    ly = df.loc[cfg_run.start_fcst - pd.DateOffset(years=1) : cfg_run.end_fcst - pd.DateOffset(years=1)]

    # --- Baseline 6-month totals ---
    ty_units     = ty["units"].sum()
    ty_sales     = ty["sales"].sum()
    ty_cogs      = ty["cogs"].sum()
    ty_gm        = ty["gm"].sum()
    ty_ni        = ty["net_income"].sum()
    ty_avg_price = ty["sales"].sum() / ty["units"].sum()
    ty_avg_cost  = ty["cogs"].sum()  / ty["units"].sum()
    ty_fixed     = ty["fixed_cost"].sum()

    ly_units     = ly["units"].sum()
    ly_sales     = ly["sales"].sum()
    ly_cogs      = ly["cogs"].sum()
    ly_gm        = ly["gm"].sum()
    ly_ni        = ly["net_income"].sum()
    ly_avg_price = ly["sales"].sum() / ly["units"].sum()
    ly_avg_cost  = ly["cogs"].sum()  / ly["units"].sum()

    # --- Baseline YOY ---
    yoy_sales     = (ty_sales     / ly_sales     - 1) * 100
    yoy_units     = (ty_units     / ly_units     - 1) * 100
    yoy_cogs      = (ty_cogs      / ly_cogs      - 1) * 100
    yoy_gm        = (ty_gm        / ly_gm        - 1) * 100
    yoy_ni        = (ty_ni        / ly_ni        - 1) * 100
    yoy_avg_price = (ty_avg_price / ly_avg_price - 1) * 100
    yoy_avg_cost  = (ty_avg_cost  / ly_avg_cost  - 1) * 100

    # --- Slider setup ---
    sales_breakeven_price = ALL_META["sales_breakeven_price"]
    units_breakeven_price = ALL_META["units_breakeven_price"]

    # Pad the lower breakeven down 20%, the higher one up 20% (either may be higher)
    bottom = min(sales_breakeven_price, units_breakeven_price) * 0.95
    top    = max(sales_breakeven_price, units_breakeven_price) * 1.05

    # Keep the current price inside the slider range
    top    = max(top, ty_avg_price)
    bottom = min(bottom, ty_avg_price)

    price_dollar_init = float(np.clip(ALL_META["expected_price_6m"], bottom, top))

    # --- Helper: slider with zero tick ---
    def _fmt_zero(fmt: str) -> str:
        if fmt in ("%g%%", "%d%%", "%.0f%%", "%.1f%%"):
            return "0%"
        try:
            return fmt % 0
        except Exception:
            return "0"

    def slider_with_zero(label, min_value, max_value, value, step, fmt="%.1f%%", key=None,
                         zero_font_px=12, line_width_px=2, line_color="#c8c8c8", top_gap_px=0):
        v = st.slider(label, min_value, max_value, value, step, format=fmt, key=key)
        zero_text = escape(_fmt_zero(fmt))
        st.markdown(
            f"""
            <div style="position:relative;height:{top_gap_px + zero_font_px + 2}px;margin:-6px 0 10px 0;">
            <div style="position:absolute;left:50%;top:0;bottom:{zero_font_px + 2}px;
                        width:{line_width_px}px;background:{line_color};border-radius:1px;"></div>
            <div style="position:absolute;left:50%;transform:translateX(-50%);
                        top:{top_gap_px}px;font-size:{zero_font_px}px;color:#666;line-height:1;">
                {zero_text}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return v

    # --- Sliders ---
    st.subheader("Scenario Inputs: Average Price, Average Cost, and Fixed Cost")
    c1, g1, c2, g2, c3 = st.columns([1, 0.25, .8, 0.25, .8])

    with c1:
        if bottom == top:
            price_dollar = st.number_input(
                "Scenario Price",
                value=float(bottom),
                step=0.01,
                format="%.2f",
                help="Pricing corridor collapsed to a single feasible price."
            )
        else:
            price_dollar = st.slider(
                "Avg Price ($)",
                min_value=float(bottom),
                max_value=float(top),
                value=price_dollar_init,
                step=0.01,
                format="$%.2f",
                key="scenario_price_dollar",
            )

    with c2:
        avg_cost_delta_pct = slider_with_zero(
            "Avg Cost change (%)", -10.0, 10.0, 0.0, 0.5, "%.1f%%", key="avg_cost", zero_font_px=14
        )
    with c3:
        fixed_cost_delta_pct = slider_with_zero(
            "Fixed costs change (%)", -10.0, 10.0, 0.0, 0.5, "%.1f%%", key="fixed_cost", zero_font_px=14
        )

    # --- Scenario computation ---
    at_default = (
        price_dollar == price_dollar_init and
        avg_cost_delta_pct == 0.0 and
        fixed_cost_delta_pct == 0.0
    )

    if at_default:
        yoy_sales_scn     = yoy_sales
        yoy_units_scn     = yoy_units
        yoy_cogs_scn      = yoy_cogs
        yoy_gm_scn        = yoy_gm
        yoy_ni_scn        = yoy_ni
        yoy_avg_price_scn = yoy_avg_price
        yoy_avg_cost_scn  = yoy_avg_cost
    else:
        i = (price_grid_df["avg_price"] - price_dollar).abs().idxmin()
        units_scn = float(price_grid_df.loc[i, "units"])

        avg_cost_scn   = ty_avg_cost * (1 + avg_cost_delta_pct / 100)
        fixed_cost_scn = ty_fixed    * (1 + fixed_cost_delta_pct / 100)

        sales_scn = units_scn * price_dollar
        cogs_scn  = units_scn * avg_cost_scn
        gm_scn    = sales_scn - cogs_scn
        ni_scn    = sales_scn - (cogs_scn + fixed_cost_scn)

        yoy_sales_scn     = (sales_scn    / ly_sales     - 1) * 100
        yoy_units_scn     = (units_scn    / ly_units     - 1) * 100
        yoy_cogs_scn      = (cogs_scn     / ly_cogs      - 1) * 100
        yoy_gm_scn        = (gm_scn       / ly_gm        - 1) * 100
        yoy_ni_scn        = (ni_scn       / ly_ni        - 1) * 100
        yoy_avg_price_scn = (price_dollar / ly_avg_price - 1) * 100
        yoy_avg_cost_scn  = (avg_cost_scn / ly_avg_cost  - 1) * 100

    # --- Dial factory ---
    def yoy_gauge(value_pct, title, lo=-10, hi=10):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(value_pct),
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": title},
            gauge={
                "axis": {"range": [lo, hi]},
                "bar": {"thickness": 0.5},
                "steps": [
                    {"range": [lo, 0], "color": "#eeeeee"},
                    {"range": [0, hi], "color": "#e8f5e9"},
                ],
                "threshold": {"line": {"width": 2}, "thickness": 0.8, "value": 0},
            }
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=220)
        return fig

    # --- Dials ---
    st.subheader("Financial Performance YOY: 6-month YOY Outlook")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.plotly_chart(yoy_gauge(yoy_sales_scn, "Sales YOY"), use_container_width=True)
    with d2:
        st.plotly_chart(yoy_gauge(yoy_units_scn, "Units YOY"), use_container_width=True)
    with d3:
        st.plotly_chart(yoy_gauge(yoy_avg_price_scn, "Avg Price YOY"), use_container_width=True)

    d4, d5, d6 = st.columns(3)
    with d4:
        st.plotly_chart(yoy_gauge(yoy_cogs_scn, "COGS YOY"), use_container_width=True)
    with d5:
        st.plotly_chart(yoy_gauge(yoy_gm_scn, "GM YOY"), use_container_width=True)
    with d6:
        st.plotly_chart(yoy_gauge(yoy_ni_scn, "Net Income YOY"), use_container_width=True)