import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import plotly.graph_objects as go

# actual vs. fit lineplot for training period
def plot_actual_fit(df, dep, fit):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[dep], label='Observed', c='blue')
    ax.plot(df.index, df[fit], label='Fit', c='red')
    ax.legend()
    fig.tight_layout()
    return fig

# Observed, fit, and holdout forecast
def plot_observed_fit_forecast_holdout(df, dep, ols_dep, fit, forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[dep], label='Observed', c='blue')
    ax.plot(df.index, df[fit], label='Fit', c='red')
    ax.plot(df.index, df[forecast], label='Forecast', c='green')
    ax.plot(df.index, df[ols_dep], label='OLS Forecast', c='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel(dep)
    ax.set_title(f"{dep} Holdout Forecast vs. Fit for Q and R")
    ax.legend()
    fig.tight_layout()
    return fig

# plot raw betas over time 
def plot_betas(betas_df, beta_cols):
    fig, ax = plt.subplots(figsize=(10, 5))

    for col in beta_cols:
        ax.plot(betas_df.index, betas_df[col], label=col)

    ax.axhline(0, color='black', lw=1)
    ax.set_title("Coefficient paths (train window)")
    ax.legend()
    fig.tight_layout()
    return fig

# plot normalized betas over time
def plot_normalized_betas(betas_df, beta_cols):
    fig, ax = plt.subplots(figsize=(10, 5))

    stack = (
        betas_df[beta_cols]
        .abs()
        .div(betas_df[beta_cols].abs().sum(axis=1), axis=0)
    )

    stack.plot.area(ax=ax, alpha=0.7)
    ax.set_title("Relative influence of drivers (Coefficients)")
    ax.set_ylabel("Share of total |coefficient|")
    ax.set_xlabel("Time")
    ax.legend(title="Variable", loc="upper left")
    fig.tight_layout()
    return fig

# Observed, fit, and forecast
def plot_observed_forecast(df, dep, fit, forecast, start_date_forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[dep], label='Observed', c='blue')
    ax.plot(df.index, df[fit], label='Fit', c='red')
    ax.plot(df.index, df[forecast], label='Forecast', c='green')

    # --- vertical forecast boundary ---
    ax.axvline(start_date_forecast, linestyle=":", color="black", linewidth=1.5)

    ax.set_xlabel('Date')
    ax.set_ylabel(dep)
    ax.set_title((f"{dep} Observed and Forecast").upper())
    ax.legend()
    fig.tight_layout()
    return fig

# Observed, fit, and forecast
def plot_observed_forecast_yoy(df, dep, fit, forecast, start_date_forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[dep], label=f'{dep}_YOY(%)', c='blue')
    ax.plot(df.index, df[fit], label='Fit YOY%', c='red')
    ax.plot(df.index, df[forecast], label='Forecast YOY%', c='green')

    # --- vertical forecast boundary ---
    ax.axvline(start_date_forecast, linestyle=":", color="black", linewidth=1.5)

    ax.set_xlabel('Date')
    ax.set_ylabel(dep)
    ax.set_title((f"{dep} Observed and Forecast YOY(%)").upper())
    ax.legend()
    fig.tight_layout()
    return fig

def plot_rolling_forecast_chart(rolling_forecast_df, color_map, title=None):
    """
    Rolling six-month forecast chart.
    Expects rolling_forecast_df indexed by date (x-axis = index).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot whatever has data; NaNs will just not render
    for col in rolling_forecast_df.columns:
        ax.plot(
            rolling_forecast_df.index,
            rolling_forecast_df[col],
            label=col,
            linewidth=2.0,
            linestyle='-',
            color=color_map.get(col, 'pink')
        )

    ax.set_xlabel('Month')
    ax.set_ylabel('Forecasted Values')
    if title:
        ax.set_title(title)

    # --- Force legend entries for all columns (even if all-NaN) ---
    legend_handles = [
        Line2D([0], [0], color=color_map.get(col, 'pink'), linewidth=2.0, linestyle='-')
        for col in rolling_forecast_df.columns
    ]
    ax.legend(legend_handles, list(rolling_forecast_df.columns))
    # ---------------------------------------------------------------

    fig.tight_layout()
    return fig

def plot_rolling_mape(results_6m, dep):
    """
    Plot the cumulative 6-month MAPE across the 19 rolling windows.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(results_6m["origin"], results_6m["mape"], marker="o", linewidth=2)

    ax.set_title(f"Rolling 6-Month MAPE for {dep}")
    ax.set_xlabel("Origin (0–18)")
    ax.set_ylabel("6-Month MAPE (%)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig

def plot_6m_distribution(sims, dep, bins=30):
    """
    Plot a 6-month LEVEL forecast probability distribution
    using simulation outcomes.
    """

    # compute histogram (probability density)
    counts, bin_edges = np.histogram(
        sims,
        bins=bins,
        density=True
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        bin_centers,
        counts,
        width=bin_width,
        alpha=0.75,
        edgecolor="black"
    )

    ax.set_title(f"{dep.upper()} — 6-Month Forecast Distribution")
    ax.set_xlabel("Forecast Outcome")
    ax.set_ylabel("Probability")

    ax.axvline(
        np.mean(sims),
        color="red",
        linestyle="--",
        linewidth=1,
        label="Mean"
    )

    ax.legend()
    fig.tight_layout()

    return fig


def plot_6m_yoy_distribution(sims, dep, bins=30):
    """
    Plot a 6-month YOY forecast probability distribution
    using simulation outcomes.
    """

    counts, bin_edges = np.histogram(
        sims,
        bins=bins,
        density=True
    )

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        bin_centers,
        counts,
        width=bin_width,
        alpha=0.75,
        edgecolor="black"
    )

    ax.set_title(f"{dep.upper()} — 6-Month YOY Forecast Distribution")
    ax.set_xlabel("YOY Forecast (%)")
    ax.set_ylabel("Probability")

    ax.axvline(
        np.mean(sims),
        color="red",
        linestyle="--",
        linewidth=1,
        label="Mean"
    )

    ax.legend()
    fig.tight_layout()

    return fig


# Residual Histogram
def plot_residual_histogram(residuals):

    """
    Outputs residual histogram
    Input are series and residual column is a series
    
    """
    fig, ax = plt.subplots(figsize = (8,5))
    ax.hist(residuals, bins=30, edgecolor="white")
    ax.set_title('Residuals Histogram')
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    return fig

# Plot input variable time series. This pulls ind vars from the control file.  Date must the index

def plot_ind_var_trends(df, var, start_date_forecast):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[var], label=var)
    ax.axvline(start_date_forecast, linestyle=":", color="black", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(var)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
    
def plot_ind_vars(df, var_list):
    """
    Time-series grid: one subplot per column in var_list.
    Expects df indexed by date.
    """
    sub = df[var_list].copy()
    n = len(var_list)
    rows = (n + 1) // 2  # 2 columns

    fig, axes = plt.subplots(nrows=rows, ncols=2, sharex=True, figsize=(15, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, col in enumerate(var_list):
        ax = axes[i]
        ax.plot(sub.index, sub[col], linewidth=2)
        ax.set_title(col, fontsize=12)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)

    # hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# OPTIMIZATION CHARTS ---------------------------------------

def plot_price_curve(df, exp_price, opt_price, var, title):
    """
    Generic avg_price curve plotter for Sales, GM, or NI.
    `var` must be one of: 'sixm_sales', 'sixm_gm', 'sixm_ni', etc.
    """
    # Sort prices for clean plotting
    df = df.sort_values('avg_price').copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Main curve
    ax.plot(df['avg_price'], df[var], linewidth=2.0, label="Six-Month Sales")

    # --- Expected avg_price point ---
    exp_idx = (df['avg_price'] - exp_price).abs().idxmin()
    x_exp, y_exp = df.loc[exp_idx, 'avg_price'], df.loc[exp_idx, var]

    # --- Optimal avg_price point ---
    opt_idx = (df['avg_price'] - opt_price).abs().idxmin()
    x_opt, y_opt = df.loc[opt_idx, 'avg_price'], df.loc[opt_idx, var]

    # --- Y limits with padding ---
    ymin, ymax = df[var].min(), df[var].max()
    ypad = 0.10 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # --- X limits with padding ---
    xmin, xmax = df['avg_price'].min(), df['avg_price'].max()
    xpad = 0.10 * (xmax - xmin)
    ax.set_xlim(xmin - xpad, xmax + xpad)

    # Vertical markers
    bottom = ax.get_ylim()[0]

    # Expected avg_price marker
    ax.vlines(x_exp, ymin=bottom, ymax=y_exp,
              color='grey', linestyles='--',
              label=f'Expected Price ({x_exp:.2f})')
    ax.scatter([x_exp], [y_exp], color='grey', s=50, zorder=3)

    # Optimal avg_price marker
    ax.vlines(x_opt, ymin=bottom, ymax=y_opt,
              color='red', linestyles='--',
              label=f'Sales Break Even Price ({x_opt:.2f})')
    ax.scatter([x_opt], [y_opt], color='red', s=60, zorder=4)

    ax.set_title(title)
    ax.set_xlabel('Average Price')
    ax.set_ylabel("Six-Month Sales")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_units_vs_price(df, exp_price, units_opt_price, title="6-Month Units vs Price"):
    """
    Plots 6-month units vs avg_price using the same high-quality formatting as plot_price_curve.
    Includes:
        - Expected avg_price marker
        - Units-optimal avg_price marker (YOY-flat target)
    """
    # Sort for clean plotting
    df = df.sort_values("avg_price").copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    # Main curve
    ax.plot(df["avg_price"], df["units"], linewidth=2.0, label="Six-Month Units")

    # --- Expected avg_price marker ---
    exp_idx = (df["avg_price"] - exp_price).abs().idxmin()
    x_exp = df.loc[exp_idx, "avg_price"]
    y_exp = df.loc[exp_idx, "units"]

    # --- Units-optimal avg_price marker ---
    opt_idx = (df["avg_price"] - units_opt_price).abs().idxmin()
    x_opt = df.loc[opt_idx, "avg_price"]
    y_opt = df.loc[opt_idx, "units"]

    # --- Dynamic Y limits ---
    ymin, ymax = df["units"].min(), df["units"].max()
    ypad = 0.10 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_ylim(ymin - ypad, ymax + ypad)

    # --- Dynamic X limits ---
    xmin, xmax = df["avg_price"].min(), df["avg_price"].max()
    xpad = 0.10 * (xmax - xmin)
    ax.set_xlim(xmin - xpad, xmax + xpad)

    # Bottom for vlines
    bottom = ax.get_ylim()[0]

    # Expected avg_price — grey
    ax.vlines(x_exp, ymin=bottom, ymax=y_exp,
              color="grey", linestyles="--",
              label=f"Expected Price ({x_exp:.2f})")
    ax.scatter([x_exp], [y_exp], color="grey", s=60, zorder=3)

    # Units-optimal avg_price — red
    ax.vlines(x_opt, ymin=bottom, ymax=y_opt,
              color="red", linestyles="--",
              label=f"Units Break Even Price ({x_opt:.2f})")
    ax.scatter([x_opt], [y_opt], color="red", s=70, zorder=4)

    ax.set_title(title)
    ax.set_xlabel("Price")
    ax.set_ylabel("Six-Month Units")
    ax.legend()
    fig.tight_layout()

    return fig

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_price_range(min_price, max_price, exp_price):

    min_price = round(min_price, 2)
    max_price = round(max_price, 2)
    exp_price = round(exp_price, 2)

    # Order the two breakevens (either may be higher)
    lo_be = min(min_price, max_price)
    hi_be = max(min_price, max_price)

    # Axis extends 20% beyond each breakeven; also keep exp_price in view
    ax_min = min(lo_be * 0.99, exp_price + 0.01)
    ax_max = max(hi_be * 1.01, exp_price + 0.01)

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.plot([ax_min, ax_max], [0, 0], color="gray", linewidth=2)
    ax.scatter([min_price, max_price, exp_price], [0, 0, 0], color=["blue", "blue", "red"], s=100, zorder=3)

    ax.text(min_price, 0.01, f"Sales Break Even\n${min_price:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(max_price, 0.01, f"Units Break Even\n${max_price:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(exp_price, -0.01, f"Expected Price\n${exp_price:.2f}", ha="center", va="top", fontsize=9)

    ax.set_xlim(ax_min, ax_max)
    ax.axis("off")
    plt.tight_layout()
    return fig
 
def scatter_by_year(x, y, x_label=None, y_label=None, title=None):
    years = x.index.year
    uniq  = sorted(years.unique())
    codes = years.map({yr: i for i, yr in enumerate(uniq)})
    cmap  = ListedColormap(list(plt.cm.tab10.colors)[:len(uniq)])

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(x, y, c=codes, cmap=cmap, s=30, alpha=0.7, edgecolors='none')

    handles, _ = sc.legend_elements(num=len(uniq))
    ax.legend(handles, [str(yr) for yr in uniq], title='Year', ncol=2, loc='upper left')
    ax.set_xlabel(x_label or getattr(x, 'name', 'x'))
    ax.set_ylabel(y_label or getattr(y, 'name', 'y'))
    if title: ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
    
# place dict before call
def market_share_by_year(df, var, label=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df.index, df[var], color="grey", alpha=0.6, label=label or var)

    # --- Rolling mean smoothing ---
    smooth = df[var].rolling(window=8, center=True, min_periods=1).mean()
    ax.plot(df.index, smooth, color="red", linewidth=2, label="Smoothed (Rolling 8)")

    ax.set_title("Market Share by Year")
    ax.legend()
    fig.tight_layout()
    return fig


import matplotlib.pyplot as plt
import numpy as np

def plot_ttm_trends(df_block, var_name, title, yaxis_title):
    df_plot = (
        df_block[df_block["var"] == var_name]
        .copy()
        .sort_values("period_date")
        .set_index("period_date")
    )

    # x positions (0..11)
    x = np.arange(len(df_plot))
    xlabels = df_plot.index.strftime("%Y-%m")

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Lines on primary axis
    ax1.plot(x, df_plot["ty"].values, marker="o", label="Current Year (TTM)")
    ax1.plot(x, df_plot["ly"].values, marker="o", label="Previous Year")
    ax1.set_ylabel(yaxis_title)
    ax1.set_title(title)

    # Bars on secondary axis (WIDE)
    ax2 = ax1.twinx()
    ax2.bar(x, df_plot["yoy"].values, width=0.85, alpha=0.3, label="YoY %")
    ax2.set_ylabel("YoY (%)")

    # Symmetric YoY axis scaling
    ymax = float(np.nanmax(np.abs(df_plot["yoy"].values)))
    cap = max(20.0, round((ymax * 2) / 5) * 5)
    ax2.set_ylim(-cap, cap)

    # X ticks as months
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, rotation=45, ha="right")
    ax1.set_xlabel("Month")

    # One combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    plt.tight_layout()
    return fig






    