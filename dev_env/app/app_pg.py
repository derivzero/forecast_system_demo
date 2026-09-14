# ----------------------------
# Load Core Folders
# ----------------------------

import streamlit as st
import os
from dotenv import load_dotenv

#load_dotenv()

# ----------------------------
# Streamlit MUST be configured first
# ----------------------------

st.set_page_config(page_title="Grocery Dashboard", layout="wide")

import time
SCRIPT_START = time.time()


# --------------------------------------
# Tell Python where the project root is
# -------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------
# app_pg.py — MINIMAL FIRST PASS
# ----------------------------------------
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import psycopg2

from py_files.pg_chart_specs import CHART_SPECS
from py_files.run_configs import RUN_CONFIG as cfg_run

# Define slices ---------------------------------------------------------
start_train = pd.to_datetime(cfg_run.start_train)
end_train   = pd.to_datetime(cfg_run.end_train)
start_fcst = pd.to_datetime(cfg_run.start_fcst)
end_fcst   = pd.to_datetime(cfg_run.end_fcst)

# ----------------------------------------
# PostgreSQL connection + query_pg
# ----------------------------------------
def query_pg(sql: str, params: tuple | None = None) -> pd.DataFrame:
    conn = psycopg2.connect(
    host=os.getenv("PG_HOST", "postgres"),
    database=os.getenv("PG_DATABASE", "postgres"),
    user="forecast_app_user",
    password=os.getenv("PG_APP_PASSWORD")
    )
    try:
        df = pd.read_sql(sql, conn, params=params)
        df.columns = [c.lower() for c in df.columns]
        return df
    finally:
        try:
            conn.close()
        except Exception:
            pass

# --------------------------------------
# Open access - public demo, no authentication gate
# --------------------------------------


# --------------------------------------------------
# Grabs dfs that are published and visible
# -------------------------------------------------
def get_available_runs() -> pd.DataFrame:
    sql = """
        SELECT run_id, publish_date, notes
        FROM meta.forecast_registry
        WHERE status = 'published'
          AND is_visible = TRUE
        ORDER BY publish_date DESC, run_id DESC
    """

    df = query_pg(sql)
    return df

# Loads charts associated with a run_id
def load_charts(run_id):
    sql = """
        SELECT chart_key, image_data
        FROM output.chart_artifacts
        WHERE run_id = %s
        ORDER BY chart_key
    """
    return query_pg(sql, (run_id,))


# ------------------------------------------------------------------------
# Grabs the artifacts associated with a run_id. Default is latest run_id
# ------------------------------------------------------------------------

# run function to call PG and grab available data
runs_df = get_available_runs()

# alert if no data loaded
if runs_df.empty:
    st.error("No published runs found.")
    st.stop()

# UI selector (defaults to latest because runs_df is ordered DESC)
runs_df["run_id"] = runs_df["run_id"].astype(str)

col1, col2, _ = st.columns([1, 2, 2])
with col1:
    st.markdown("<div style='font-size:20px; font-weight:600;'>Forecast run</div>", unsafe_allow_html=True)
    run_id = st.selectbox(
        label="forecast_run_hidden",
        options=runs_df["run_id"].tolist(),
        index=0,
        label_visibility="collapsed",
    )

selected_note = runs_df.loc[runs_df["run_id"] == run_id, "notes"].iloc[0]

with col2:
    if selected_note:
        st.markdown("<div style='font-size:20px; font-weight:600;'>Scenario</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px;'>{selected_note}</div>", unsafe_allow_html=True)

# call to load charts associated with the run_id selectbox
charts_df = load_charts(run_id)

if charts_df.empty:
    st.warning("No charts found for this run.")
    st.stop()

# Converts the charts_df chart_key and image data into a dict.  Zip associates the key (chart_key) and value (image_data)
chart_map = dict(zip(charts_df["chart_key"], charts_df["image_data"]))

# ------------------------------------------------------------------------
# Chart render function
# ------------------------------------------------------------------------
def render(chart_key, col):
    img = chart_map.get(chart_key)

    if img is None:
        col.error(f"Missing chart: {chart_key}")
        return

    # Normalize PostgreSQL BYTEA types
    if isinstance(img, memoryview):
        img = img.tobytes()
    elif isinstance(img, bytearray):
        img = bytes(img)

    col.image(img, use_container_width=True)

# ------------------------------------------------------------------------
# Dial functions
# ------------------------------------------------------------------------
def yoy_gauge(value, vmin=-20, vmax=20, height=200, top_margin=30):
    """
    Standard YOY gauge dial. Expects value in percent units (e.g., 3.4 for 3.4%).
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={"axis": {"range": [vmin, vmax]}}
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=top_margin, b=10)
    )
    return fig

def plot_elasticity_dial(E_rel):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(E_rel),
        title={"text": "Price Elasticity"},
        gauge={"axis": {"range": [-3, 0]}}
    ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=80, b=10),
        height=400
    )

    return fig

def dial_header(txt, size=16):
    st.markdown(
        f"<h3 style='text-align:center; font-size:{size}px; margin:0'>{txt}</h3>",
        unsafe_allow_html=True
    )

def get_dial_value(run_id, metric, value_type):
    sql = """
        SELECT value
        FROM output.dial_values
        WHERE run_id = %s
          AND metric = %s
          AND value_type = %s
    """

    df = query_pg(sql, (run_id, metric, value_type))

    if df.empty:
        raise ValueError(
            f"No dial value for metric={metric}, value_type={value_type}, run={run_id}"
        )

    return float(df["value"].iloc[0])


# ---- Dial factory ----
def yoy_gauge_scenarios(value_pct, baseline_pct, title, lo=-10, hi=10):
    if value_pct is None:
        return go.Figure()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value_pct),
        number={"suffix": "%", "valueformat": ".1f"},
        delta={"reference": (baseline_pct if baseline_pct is not None else 0.0),
            "relative": False, "valueformat": ".1f", "suffix": "%"},
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

# ------------------------------------------------------------------------
# Slider functions
# ------------------------------------------------------------------------
# This function allows for the creation of the interactive sliders
def scenario_forward_only(
    base,                      # DataFrame with columns listed below
    start_fcst,                # pd.Timestamp
    pct_price=0.0,             # Δ% for avg_price
    pct_avg_cost=0.0,          # Δ% for avg_cost (variable/unit cost)
    pct_fixed=0.0,             # Δ% for fixed cost
    elasticity=0.0,            # units response to avg_price; e.g., -1.2
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

    # Adjusted avg_price & avg_cost (forecast horizon only)
    out["price_adj"] = out[price_col]
    out.loc[mask, "price_adj"] = out.loc[mask, price_col] * mp

    out["avg_cost_adj"] = out[cost_col]
    out.loc[mask, "avg_cost_adj"] = out.loc[mask, cost_col] * mc

    # Adjusted fixed (only used if include_fixed_in_gm=True)
    out["fixed_adj"] = out[fixed_col] if fixed_col in out.columns else 0.0
    if fixed_col in out.columns:
        out.loc[mask, "fixed_adj"] = out.loc[mask, fixed_col] * mfix
    else:
        out["fixed_adj"] = 0.0

    # Units (optionally react to avg_price)
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

# --- Helper: slider with zero tick (used for cost sliders) ---
def _fmt_zero(fmt: str) -> str:
    if fmt in ("%g%%", "%d%%", "%.0f%%", "%.1f%%"):
        return "0%"
    try:
        return fmt % 0
    except Exception:
        return "0"

def slider_with_zero(label, min_value, max_value, value, step, fmt="%g%%", key=None,
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

# ----------------------------------------------------------------------
# Config to clean up and standardize charts
# ----------------------------------------------------------------------

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

# ---------------------------------------------------------------
# Sidebar Nav: two grouped radios, mutually exclusive
# --------------------------------------------------------------

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
    index=0,
    key="market_nav",
    label_visibility="collapsed",          # hides it visually, keeps accessibility
    on_change=choose_market,
)

st.sidebar.subheader("Retailer History (TTM)")
st.sidebar.radio(
    label="Retailer Navigation TTM",           # non-empty!
    options=["KPI Snapshot TTM", "KPI Trends TTM", "KPI Tree TTM"],
    index=0,
    key="retailer_ttm_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_ttm,
)

st.sidebar.subheader("Retailer Demand Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Demand Forecast",           # non-empty!
    options=["Unit Forecast", "Sales Forecast", "Visit Forecast", "UPV Forecast"],
    index=0,
    key="retailer_demand_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_demand_fcst,
)

st.sidebar.subheader("Retailer Cost Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Cost Forecast",           # non-empty!
    options=["Average Cost Forecast", "COGS Forecast", "Total Cost Forecast"],
    index=0,
    key="retailer_cost_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_cost_fcst,
)

st.sidebar.subheader("Retailer Margin Forecasts")
st.sidebar.radio(
    label="Retailer Navigation Margin Forecast",           # non-empty!
    options=["Gross Margin Forecast", "Net Income Forecast", "KPI Tree Forecast"],
    index=0,
    key="retailer_margin_fcst_nav",
    label_visibility="collapsed",
    on_change=choose_retailer_margin_fcst,
)

st.sidebar.subheader("Optimization")
st.sidebar.radio(
    label="Optimization",           # non-empty!
    options=["Gross Margin, Net Income, and Market Share Trends", "Financial Simulator"],
    index=0,
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

    IMG_DIR = Path(__file__).resolve().parent / "tab0"
    logo_path = IMG_DIR / "dz_logo.jpg"

    left, r1c1, right = st.columns([0.2, 0.4, 0.45])
    r1c1.image(logo_path.read_bytes(), use_container_width=True)

# -----------------------------------------------------------------------------------------------
# TAB 1 - Market Trends
# -----------------------------------------------------------------------------------------------
elif tab == "Market Charts TTM":

    st.header("U.S. Grocery Market Data: Trailing 12 Month (TTM)")

    # ----------------------------------------
    # Load TTM YOY dials from Snowflake
    # ----------------------------------------
    
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('cpi_fah', 'units_mkt_trend', 'sales_mkt_trend')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # ----------------------------------------
    # Row 1: YOY TTM Dials
    # ----------------------------------------
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        st.markdown("### U.S. CPI FAH TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cpi_fah"]), use_container_width=True)

    with r1c2:
        st.markdown("### U.S. Grocery Market Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units_mkt_trend"]), use_container_width=True)

    with r1c3:
        st.markdown("### U.S. Grocery Market Sales TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales_mkt_trend"]), use_container_width=True)

       
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        render("cpi_fah_yoy", r2c1)
    with r2c2:
        render("units_mkt_trend_yoy", r2c2)
    with r2c3:
        render("sales_mkt_trend_yoy", r2c3)


# -------------------------------------------------------------------------------------
# TAB 2 - CPI FAH Forecast
# -------------------------------------------------------------------------------------
elif tab == "CPI FAH Forecast":

    st.header("CPI Food-at-Home")

    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('cpi_fah', 'oil_prices_lag7', 'ppi_farm_products_lag4', 'ppi_food_mfg_lag3', 'ppi_grocery')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'cpi_fah'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))


    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### CPI FAH TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cpi_fah"]), use_container_width=True)

    with r1c2:
        st.markdown("### CPI FAH 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["cpi_fah"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r2c1:
        st.markdown("### Oil Prices YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["oil_prices_lag7"]), use_container_width=True)

    with r2c2:
        st.markdown("### PPI Farm Products YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_farm_products_lag4"]), use_container_width=True)

    with r2c3:
        st.markdown("### PPI Food Manufacture", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_food_mfg_lag3"]), use_container_width=True)

    with r2c4:
        st.markdown("### PPI Grocery Retail", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_grocery"]), use_container_width=True)

    
    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1:
        render("oil_prices_lag7_yoy", r3c1)
    with r3c2:
        render("ppi_farm_products_lag4_yoy", r3c2)
    with r3c3:
        render("ppi_food_mfg_lag3_yoy", r3c3)
    with r3c4:
        render("ppi_grocery_yoy", r3c4)
    
    st.divider()
    # Row 4: Betas -------------------------
    st.subheader("Driver Importance")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("cpi_fah_betas", r4c1)
    with r4c2:
        render("cpi_fah_betas_normalized", r4c2)
    
    st.divider()
    # Row 5: Forecasts -------------------------
    
    st.subheader("Forecast: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("cpi_fah_level_forecast", r5c1)
    with r5c2:
        render("cpi_fah_level_forecast_yoy", r5c2)

    st.divider()
    # Row 6: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        render("cpi_fah_forecast_distribution", r6c1)
    with r6c2:
        render("cpi_fah_forecast_distribution_yoy", r6c2)
    
    st.divider()
    #  Row 7: Forecast Performance -----------------------------
    st.subheader("Holdout Analysis and Model Performance")
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        render("cpi_fah_forecast_holdout", r7c1)
    with r7c2:
        render("cpi_fah_rolling_6m_mape_holdout", r7c2)

# ----------------------------------------------------------------------------------------
# TAB 3 - US Units
# ----------------------------------------------------------------------------------------
elif tab == "US Units Forecast":
    st.header("US Units")

    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('units_mkt_trend', 'avg_price_trend', 'rdi', 'home_price')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'units_mkt_trend'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    
    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### US Units YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units_mkt_trend"]), use_container_width=True)

    with r1c2:
        st.markdown("### US Units 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["units_mkt_trend"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        st.markdown("### U.S. CPI FAH TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price_trend"]), use_container_width=True)

    with r2c2:
        st.markdown("### U.S. RDI TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["rdi"]), use_container_width=True)

    with r2c3:
        st.markdown("### U.S. HOME PRICE TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["home_price"]), use_container_width=True)    

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        render("cpi_fah_yoy", r3c1)
    with r3c2:
        render("rdi_yoy", r3c2)
    with r3c3:
        render("home_price_yoy", r3c3)
    
    st.divider()
    # Row 4: Betas -------------------------
    st.subheader("Driver Importance")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("units_mkt_trend_betas", r4c1)
    with r4c2:
        render("units_mkt_trend_betas_normalized", r4c2)
    
    st.divider()
    # Row 5: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("units_mkt_trend_level_forecast", r5c1)
    with r5c2:
        render("units_mkt_trend_level_forecast_yoy", r5c2)
        
    
    st.divider()
    # Row 6: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        render("units_mkt_trend_forecast_distribution", r6c1)
    with r6c2:
        render("units_mkt_trend_forecast_distribution_yoy", r6c2)
    
    st.divider()
    #  Row 7: Forecast Performance -----------------------------
    st.subheader("Holdout Analysis and Model Performance")
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        render("units_mkt_trend_forecast_holdout", r7c1)
    with r7c2:
        render("units_mkt_trend_rolling_6m_mape_holdout", r7c2)
    
# -------------------------------------------------------------------------------------
# TAB 4 - US Sales Forecast
# -------------------------------------------------------------------------------------
elif tab == "US Sales Forecast":
    st.header("US Sales")
    
 # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('sales_mkt_trend', 'units_mkt_trend', 'avg_price_trend')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'sales_mkt_trend'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### US Sales YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales_mkt_trend"]), use_container_width=True)

    with r1c2:
        st.markdown("### US Sales 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["sales_mkt_trend"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### U.S. CPI FAH TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price_trend"]), use_container_width=True)

    with r2c2:
        st.markdown("### U.S. Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units_mkt_trend"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("avg_price_trend_yoy", r3c1)
    with r3c2:
        render("units_mkt_trend_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("sales_mkt_trend_level_forecast", r4c1)
    with r4c2:
        render("sales_mkt_trend_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("sales_mkt_trend_forecast_distribution", r5c1)
    with r5c2:
        render("sales_mkt_trend_forecast_distribution_yoy", r5c2)
# ----------------------------------------------------------------------------------------
# TAB 5 - KPI Snapshot
# ----------------------------------------------------------------------------------------
elif tab == "KPI Snapshot TTM":
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('units', 'avg_price', 'avg_cost', 'sales', 'cogs', 'gm')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Row 1: YOY TTM Dials ------------------------
    r1c1, = st.columns(1)
    with r1c1:
        st.markdown("### Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)
    
    st.divider()
    # row 2 ----------------------------------
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("### Avg Price TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price"]), use_container_width=True)
    with r2c2:
        st.markdown("### Avg Cost TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_cost"]), use_container_width=True)
    
    st.divider()
    # row 3 ----------------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.markdown("### Sales TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales"]), use_container_width=True)
    with r3c2:
        st.markdown("### COGS TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cogs"]), use_container_width=True)
    
    st.divider()
    # Row 4 -----------------------------------
    r4c1, = st.columns(1)
    with r4c1:
        st.markdown("### Gross Margin TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["gm"]), use_container_width=True)
        
# ----------------------------------------------------------------------------------------
# TAB 6 - KPI Trends
# ----------------------------------------------------------------------------------------
elif tab == "KPI Trends TTM":

    st.header("KPI Trends: TTM vs Prior Year")
    r1c1, = st.columns(1)
    with r1c1:
        render("units_ttm_comp", r1c1)
    
    st.divider()
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        render("avg_price_ttm_comp", r2c1)
    with r2c2:
        render("avg_cost_ttm_comp", r2c2)
        
    st.divider()
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("sales_ttm_comp", r3c1)
    with r3c2:
        render("cogs_ttm_comp", r3c2)
    
    st.divider()
    r4c1, = st.columns(1)
    with r4c1:
        render("gm_ttm_comp", r4c1)
    
# -------------------------------------------------------------------------------------
# TAB 7 - KPI Tree TTM
# -------------------------------------------------------------------------------------
elif tab == "KPI Tree TTM":
    from py_files.utils import decomp_sales_bennett, decomp_cogs_bennett

    st.header("KPI Tree: Trailing 12 Months (TTM)")

    # -------------------------------------------------
    # Pull pre-aggregated KPI snapshot from Snowflake
    # -------------------------------------------------
    df_tree = query_pg("""
    SELECT *
    FROM output.v_tree_ttm
    WHERE run_id = %s
    """, (run_id,))

    df_tree = df_tree.set_index("var")

    # -------------------------------------------------
    # Extract scalars
    # -------------------------------------------------

    units_ty  = float(df_tree.loc["units", "ty"])
    units_ly  = float(df_tree.loc["units", "ly"])
    units_yoy = float(df_tree.loc["units", "yoy"])

    sales_ty  = float(df_tree.loc["sales", "ty"])
    sales_ly  = float(df_tree.loc["sales", "ly"])
    sales_yoy = float(df_tree.loc["sales", "yoy"])

    cogs_ty   = float(df_tree.loc["cogs", "ty"])
    cogs_ly   = float(df_tree.loc["cogs", "ly"])
    cogs_yoy  = float(df_tree.loc["cogs", "yoy"])

    gm_yoy         = float(df_tree.loc["gm", "yoy"])
    avg_price_yoy  = float(df_tree.loc["avg_price", "yoy"])
    avg_cost_yoy   = float(df_tree.loc["avg_cost", "yoy"])

    dS = float(df_tree.loc["sales", "diff"])
    dC = float(df_tree.loc["cogs", "diff"])
    dG = float(df_tree.loc["gm", "diff"])

    # -------------------------------------------------
    # Bennett decomposition (Python-side, scalar-safe)
    # -------------------------------------------------
    dS_price, dS_units = decomp_sales_bennett(
        ly_units=units_ly,
        ly_sales=sales_ly,
        ty_units=units_ty,
        ty_sales=sales_ty,
    )

    dC_cost, dC_units = decomp_cogs_bennett(
        ly_units=units_ly,
        ly_cogs=cogs_ly,
        ty_units=units_ty,
        ty_cogs=cogs_ty,
    )

    # -------------------------------------------------
    # Formatting helpers (strings only, no DF mutation)
    # -------------------------------------------------
    def fmt_money(x):
        ax = abs(x)
        if ax >= 1e9: return f"${x/1e9:.1f}B"
        if ax >= 1e6: return f"${x/1e6:.1f}M"
        if ax >= 1e3: return f"${x/1e3:.0f}K"
        return f"${x:.0f}"

    def signed(x):
        return ("+" if x >= 0 else "−") + fmt_money(abs(x))

    # -------------------------------------------------
    # Graphviz DOT
    # -------------------------------------------------
    dot = f"""
    digraph {{
      rankdir=TB;
      nodesep=0.75; ranksep=0.35;
      node [shape=box, style="rounded,filled", color="#cccccc", fillcolor="white", fontsize=12];

      GM  [label="GM$ YoY\\n{gm_yoy:.1f}%\\nΔ {signed(dG)}"];
      S   [label="Sales YoY\\n{sales_yoy:.1f}%\\nΔ {signed(dS)}"];
      C   [label="COGS YoY\\n{cogs_yoy:.1f}%\\nΔ {signed(dC)}"];

      U1  [label="Units YoY\\n{units_yoy:.1f}%\\nΔ {signed(dS_units)}"];
      P   [label="Avg Price YoY\\n{avg_price_yoy:.1f}%\\nΔ {signed(dS_price)}"];

      U2  [label="Units YoY\\n{units_yoy:.1f}%\\nΔ {signed(dC_units)}"];
      AC  [label="Avg Cost YoY\\n{avg_cost_yoy:.1f}%\\nΔ {signed(dC_cost)}"];

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
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('units', 'avg_price', 'rdi', 'home_price')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'units'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    
    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Units YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)

    with r1c2:
        st.markdown("### Units 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["units"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        st.markdown("### Avg Price TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price"]), use_container_width=True)

    with r2c2:
        st.markdown("### U.S. RDI TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["rdi"]), use_container_width=True)

    with r2c3:
        st.markdown("### U.S. HOME PRICE TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["home_price"]), use_container_width=True)    

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        render("avg_price_yoy", r3c1)
    with r3c2:
        render("rdi_yoy", r3c2)
    with r3c3:
        render("home_price_yoy", r3c3)
    
    st.divider()
    # Row 4: Betas -------------------------
    st.subheader("Driver Importance")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("units_betas", r4c1)
    with r4c2:
        render("units_betas_normalized", r4c2)
    
    st.divider()
    # Row 5: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("units_level_forecast", r5c1)
    with r5c2:
        render("units_level_forecast_yoy", r5c2)

    st.divider()
    # Row 6: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        render("units_forecast_distribution", r6c1)
    with r6c2:
        render("units_forecast_distribution_yoy", r6c2)
    
    st.divider()
    #  Row 7: Forecast Performance -----------------------------
    st.subheader("Holdout Analysis and Model Performance")
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        render("units_forecast_holdout", r7c1)
    with r7c2:
        render("units_rolling_6m_mape_holdout", r7c2)

# -------------------------------------------------------------------------------------
# TAB 9 - Sales Forecast
# -------------------------------------------------------------------------------------
elif tab == "Sales Forecast":
    st.header("Retailer Sales")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('sales', 'units', 'avg_price')
    """, (run_id))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'sales'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### US Sales YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales"]), use_container_width=True)

    with r1c2:
        st.markdown("### US Sales 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["sales"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### U.S. CPI FAH TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price"]), use_container_width=True)

    with r2c2:
        st.markdown("### U.S. Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("avg_price_yoy", r3c1)
    with r3c2:
        render("units_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("sales_level_forecast", r4c1)
    with r4c2:
        render("sales_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("sales_forecast_distribution", r5c1)
    with r5c2:
        render("sales_forecast_distribution_yoy", r5c2)
    
# -------------------------------------------------------------------------------------
# TAB 10 - Visits Forecast
# -------------------------------------------------------------------------------------
elif tab == "Visit Forecast":
    st.header("Retailer Visits")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('visits', 'avg_price', 'units')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id, 'visits'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    
    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Units YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["visits"]), use_container_width=True)

    with r1c2:
        st.markdown("### Units 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["visits"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### Avg Price TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_price"]), use_container_width=True)

    with r2c2:
        st.markdown("### Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("avg_price_yoy", r3c1)
    with r3c2:
        render("units_yoy", r3c2)
       
    st.divider()
    # Row 4: Betas -------------------------
    st.subheader("Driver Importance")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("visits_betas", r4c1)
    with r4c2:
        render("visits_betas_normalized", r4c2)
    
    st.divider()
    # Row 5: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("visits_level_forecast", r5c1)
    with r5c2:
        render("visits_level_forecast_yoy", r5c2)
    
    st.divider()
    # Row 6: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        render("visits_forecast_distribution", r6c1)
    with r6c2:
        render("visits_forecast_distribution_yoy", r6c2)
    
    st.divider()
    #  Row 7: Forecast Performance -----------------------------
    st.subheader("Holdout Analysis and Model Performance")
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        render("visits_forecast_holdout", r7c1)
    with r7c2:
        render("visits_rolling_6m_mape_holdout", r7c2)

# -------------------------------------------------------------------------------------
# TAB 11 - UPV Forecast
# -------------------------------------------------------------------------------------
elif tab == "UPV Forecast":
    st.header("Retailer Units per Visit")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('upv', 'units', 'visits')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'upv'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Units Per Visit YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["upv"]), use_container_width=True)

    with r1c2:
        st.markdown("### Units Per Visit 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["upv"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)

    with r2c2:
        st.markdown("### Visits TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["visits"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("units_yoy", r3c1)
    with r3c2:
        render("visits_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("upv_level_forecast", r4c1)
    with r4c2:
        render("upv_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("upv_forecast_distribution", r5c1)
    with r5c2:
        render("upv_forecast_distribution_yoy", r5c2)

# -------------------------------------------------------------------------------------
# TAB 12 - Avg Cost Forecast
# -------------------------------------------------------------------------------------
elif tab == "Average Cost Forecast":
    st.header("Retailer Average Cost")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('avg_cost', 'ppi_food_mfg_lag2', 'ppi_food_mfg_lag3', 'ppi_food_mfg_lag4')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'avg_cost'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    
    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Units YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_cost"]), use_container_width=True)

    with r1c2:
        st.markdown("### Units 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["avg_cost"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        st.markdown("### PPI Food Manufacture L2 TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_food_mfg_lag2"]), use_container_width=True)

    with r2c2:
        st.markdown("### PPI Food Manufacture L3 TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_food_mfg_lag3"]), use_container_width=True)

    with r2c3:
        st.markdown("### PPI Food Manufacture L4 TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["ppi_food_mfg_lag4"]), use_container_width=True)    

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        render("ppi_food_mfg_lag2_yoy", r3c1)
    with r3c2:
        render("ppi_food_mfg_lag3_yoy", r3c2)
    with r3c3:
        render("ppi_food_mfg_lag4_yoy", r3c3)
    
    st.divider()
    # Row 4: Betas -------------------------
    st.subheader("Driver Importance")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("avg_cost_betas", r4c1)
    with r4c2:
        render("avg_cost_betas_normalized", r4c2)
    
    
    st.divider()
    # Row 5: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("avg_cost_level_forecast", r5c1)
    with r5c2:
        render("avg_cost_level_forecast_yoy", r5c2)
    
    st.divider()
    # Row 6: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        render("avg_cost_forecast_distribution", r6c1)
    with r6c2:
        render("avg_cost_forecast_distribution_yoy", r6c2)
    
    st.divider()
    #  Row 7: Forecast Performance -----------------------------
    st.subheader("Holdout Analysis and Model Performance")
    r7c1, r7c2 = st.columns(2)
    with r7c1:
        render("avg_cost_forecast_holdout", r7c1)
    with r7c2:
        render("avg_cost_rolling_6m_mape_holdout", r7c2)

# -------------------------------------------------------------------------------------
# TAB 13 - COGS Forecast
# -------------------------------------------------------------------------------------
elif tab == "COGS Forecast":
    st.header("Retailer Cost of Goods Sold")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('cogs', 'avg_cost', 'units')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'cogs'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### COGS TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cogs"]), use_container_width=True)

    with r1c2:
        st.markdown("### COGS 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["cogs"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### Units TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["units"]), use_container_width=True)

    with r2c2:
        st.markdown("### Avg Cost TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["avg_cost"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("units_yoy", r3c1)
    with r3c2:
        render("avg_cost_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("cogs_level_forecast", r4c1)
    with r4c2:
        render("cogs_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("cogs_forecast_distribution", r5c1)
    with r5c2:
        render("cogs_forecast_distribution_yoy", r5c2)


# -------------------------------------------------------------------------------------
# TAB 14 - Total Cost Forecast
# -------------------------------------------------------------------------------------
elif tab == "Total Cost Forecast":
    st.header("Retailer Total Cost")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('total_cost', 'cogs', 'fixed_cost')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'total_cost'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Total Cost TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["total_cost"]), use_container_width=True)

    with r1c2:
        st.markdown("### Total Cost 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["total_cost"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### COGS TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cogs"]), use_container_width=True)

    with r2c2:
        st.markdown("### Fixed Cost TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["fixed_cost"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("cogs_yoy", r3c1)
    with r3c2:
        render("fixed_cost_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("total_cost_level_forecast", r4c1)
    with r4c2:
        render("total_cost_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("total_cost_forecast_distribution", r5c1)
    with r5c2:
        render("total_cost_forecast_distribution_yoy", r5c2)

# -------------------------------------------------------------------------------------
# TAB 15 - GM Forecast
# -------------------------------------------------------------------------------------
elif tab == "Gross Margin Forecast":
    st.header("Retailer Gross Margin")
    
    # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('gm', 'sales', 'cogs')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'gm'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Gross Margin TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["gm"]), use_container_width=True)

    with r1c2:
        st.markdown("### Gross Margin 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["gm"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### Sales TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales"]), use_container_width=True)

    with r2c2:
        st.markdown("### COGS TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["cogs"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("sales_yoy", r3c1)
    with r3c2:
        render("cogs_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("gm_level_forecast", r4c1)
    with r4c2:
        render("gm_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("gm_forecast_distribution", r5c1)
    with r5c2:
        render("gm_forecast_distribution_yoy", r5c2)


# -------------------------------------------------------------------------------------
# TAB 16 - NI Forecast
# -------------------------------------------------------------------------------------
elif tab == "Net Income Forecast":
    st.header("Retailer Net Income")
    
        # Dial Views TTM --------------------------------------
    dials_ttm = query_pg("""
        SELECT var, value
        FROM output.v_dial_ttm_yoy
        WHERE run_id = %s
          AND var IN ('net_income', 'sales', 'total_cost')
    """, (run_id,))

    dials_ttm.columns = [c.lower() for c in dials_ttm.columns]
    dials_ttm = dict(zip(dials_ttm["var"], dials_ttm["value"]))

    # Dial Views Forward ----------------------------------
    dials_forward = query_pg("""
        SELECT var, value
        FROM output.v_dial_forecast_yoy
        WHERE run_id = %s AND var = %s
    """, (run_id,'net_income'))

    dials_forward.columns = [c.lower() for c in dials_forward.columns]
    dials_forward = dict(zip(dials_forward["var"], dials_forward["value"]))

    # Row 1: Dependent Var Dials ----------------------
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown("### Net Income TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["net_income"]), use_container_width=True)

    with r1c2:
        st.markdown("### Net Income 6 Month Outlook YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_forward["net_income"]), use_container_width=True)
    
    st.divider()
    # Row 2: Independent Var Dials --------------------
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown("### Sales TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["sales"]), use_container_width=True)
    with r2c2:
        st.markdown("### Total Cost TTM YOY", unsafe_allow_html=True)
        st.plotly_chart(yoy_gauge(dials_ttm["total_cost"]), use_container_width=True)

    # Row 3: Inv Var Time Series Trends YOY -------------------------
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        render("sales_yoy", r3c1)
    with r3c2:
        render("total_cost_yoy", r3c2)
    
    st.divider()
    # Row 4: Forecasts -------------------------
    st.subheader("Forecast: 6-month Outlook")
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        render("net_income_level_forecast", r4c1)
    with r4c2:
        render("net_income_level_forecast_yoy", r4c2)
    
    st.divider()
    # Row 5: Forecast Distibutions -------------------------
    st.subheader("Forecast Distributions: 6-month Outlook")
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        render("net_income_forecast_distribution", r5c1)
    with r5c2:
        render("net_income_forecast_distribution_yoy", r5c2)

# -------------------------------------------------------------------------------------
# TAB 17 - KPI Tree Forecast
# -------------------------------------------------------------------------------------

elif tab == "KPI Tree Forecast":
    import pandas as pd
    from pathlib import Path
    from py_files.utils import decomp_sales_bennett, decomp_cogs_bennett

    st.header("KPI Tree Forecast: 6-month Outlook")

    # -------------------------------------------------
    # Pull pre-aggregated KPI snapshot from Snowflake
    # -------------------------------------------------
    df_tree = query_pg("""
    SELECT *
    FROM output.v_tree_fcst
    WHERE run_id = %s
    """, (run_id,))

    df_tree = df_tree.set_index("var")

    # -------------------------------------------------
    # Extract scalars
    # -------------------------------------------------

    units_ty  = float(df_tree.loc["units", "ty"])
    units_ly  = float(df_tree.loc["units", "ly"])
    units_yoy = float(df_tree.loc["units", "yoy"])

    sales_ty  = float(df_tree.loc["sales", "ty"])
    sales_ly  = float(df_tree.loc["sales", "ly"])
    sales_yoy = float(df_tree.loc["sales", "yoy"])

    cogs_ty   = float(df_tree.loc["cogs", "ty"])
    cogs_ly   = float(df_tree.loc["cogs", "ly"])
    cogs_yoy  = float(df_tree.loc["cogs", "yoy"])

    gm_yoy         = float(df_tree.loc["gm", "yoy"])
    avg_price_yoy  = float(df_tree.loc["avg_price", "yoy"])
    avg_cost_yoy   = float(df_tree.loc["avg_cost", "yoy"])

    dS = float(df_tree.loc["sales", "diff"])
    dC = float(df_tree.loc["cogs", "diff"])
    dG = float(df_tree.loc["gm", "diff"])

    # -------------------------------------------------
    # Bennett decomposition (Python-side, scalar-safe)
    # -------------------------------------------------
    dS_price, dS_units = decomp_sales_bennett(
        ly_units=units_ly,
        ly_sales=sales_ly,
        ty_units=units_ty,
        ty_sales=sales_ty,
    )

    dC_cost, dC_units = decomp_cogs_bennett(
        ly_units=units_ly,
        ly_cogs=cogs_ly,
        ty_units=units_ty,
        ty_cogs=cogs_ty,
    )

    # -------------------------------------------------
    # Formatting helpers (strings only, no DF mutation)
    # -------------------------------------------------
    def fmt_money(x):
        ax = abs(x)
        if ax >= 1e9: return f"${x/1e9:.1f}B"
        if ax >= 1e6: return f"${x/1e6:.1f}M"
        if ax >= 1e3: return f"${x/1e3:.0f}K"
        return f"${x:.0f}"

    def signed(x):
        return ("+" if x >= 0 else "−") + fmt_money(abs(x))

    # -------------------------------------------------
    # Graphviz DOT
    # -------------------------------------------------
    dot = f"""
    digraph {{
      rankdir=TB;
      nodesep=0.75; ranksep=0.35;
      node [shape=box, style="rounded,filled", color="#cccccc", fillcolor="white", fontsize=12];

      GM  [label="GM$ YoY\\n{gm_yoy:.1f}%\\nΔ {signed(dG)}"];
      S   [label="Sales YoY\\n{sales_yoy:.1f}%\\nΔ {signed(dS)}"];
      C   [label="COGS YoY\\n{cogs_yoy:.1f}%\\nΔ {signed(dC)}"];

      U1  [label="Units YoY\\n{units_yoy:.1f}%\\nΔ {signed(dS_units)}"];
      P   [label="Avg Price YoY\\n{avg_price_yoy:.1f}%\\nΔ {signed(dS_price)}"];

      U2  [label="Units YoY\\n{units_yoy:.1f}%\\nΔ {signed(dC_units)}"];
      AC  [label="Avg Cost YoY\\n{avg_cost_yoy:.1f}%\\nΔ {signed(dC_cost)}"];

      U1 -> S;  P  -> S;  S -> GM;
      U2 -> C;  AC -> C;  C -> GM;
    }}
    """

    st.graphviz_chart(dot, use_container_width=True)

# -------------------------------------------------------------------------------------
# TAB 18 - Long Run charts
# -------------------------------------------------------------------------------------

elif tab == "Gross Margin, Net Income, and Market Share Trends":
    
    # Row 1: Forecast Distibutions -------------------------
    st.subheader("Gross Margin, Net Income, and Market Share Trends")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        render("avg_price_and_gm", r1c1)
    with r1c2:
        render("avg_price_and_net_income", r1c2)

    st.divider()
    st.subheader("Market Share")
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        render("market_share_sales", r2c1)
    with r2c2:
        render("market_share_units", r2c2)

# -------------------------------------------------------------------------------------
# TAB 19 - Financial Simulator
# -------------------------------------------------------------------------------------

elif tab == "Financial Simulator":
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    from html import escape

    st.header("Financial Performance Optimization: 6-month Outlook")

    # --- Load meta ---
    def load_all_meta(run_id):
        df = query_pg("""
            SELECT meta_key, meta_value
            FROM meta.meta_registry
            WHERE run_id = %s
        """, [run_id])
        df.columns = [c.lower() for c in df.columns]
        meta = {}
        for _, r in df.iterrows():
            meta[r["meta_key"]] = float(r["meta_value"])
        return meta

    ALL_META = load_all_meta(run_id)
    E_rel = ALL_META["units_E_rel"]

    # --- Elasticity dial ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=E_rel,
        title={"text": "Price Elasticity"},
        gauge={"axis": {"range": [-3, 0]}}
    ))
    left, r1c1, right = st.columns([0.3, 1, 0.3])
    with r1c1:
        st.plotly_chart(fig, use_container_width=True)

    # --- price range -----
    st.divider()
    st.subheader("Sales and Unit Break Even: 6-month Outlook")
    
    left, r3c1, right = st.columns([0.1, 1, 0.1])
    with r3c1:
        render("price_range", r3c1)
        
    st.divider()
    st.header("Scenario Planning")

    # --- Load df_wide ---
    df_wide = query_pg("""
        SELECT period_date, units, avg_price, avg_cost, fixed_cost, sales, cogs, gm, net_income
        FROM output.df_wide
        WHERE run_id = %s
        ORDER BY period_date
    """, [run_id])
    df_wide["period_date"] = pd.to_datetime(df_wide["period_date"])
    df_wide = df_wide.set_index("period_date")

    # --- Load price_grid ---
    price_grid_df = query_pg("""
        SELECT avg_price, units
        FROM output.price_optimization_table
        WHERE run_id = %s
        ORDER BY avg_price
    """, [run_id])

    # --- Parse forecast window from run_id ---
    run_id_base = run_id.split(".")[0]
    year, month = int(run_id_base.split("_")[0]), int(run_id_base.split("_")[1])
    fcst_start = pd.Timestamp(year=year, month=month, day=1)
    fcst_end   = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=5)

    # --- Forecast and LY windows ---
    ty = df_wide.loc[fcst_start : fcst_end]
    ly = df_wide.loc[fcst_start - pd.DateOffset(years=1) : fcst_end - pd.DateOffset(years=1)]

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
    bottom = min(sales_breakeven_price, units_breakeven_price) * 0.8
    top    = max(sales_breakeven_price, units_breakeven_price) * 1.2

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
    c1, g1, c2, g2, c3 = st.columns([1, 0.75, .8, 0.75, .8])

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