import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_charts(df, MODEL_CONFIG, upstream_sims_dict, downstream_sims_dict):
    
    from py_files.run_configs import RUN_CONFIG as cfg_run#

    from py_files.charts_helpers import (
    plot_observed_forecast,
    plot_observed_forecast_yoy,
    plot_6m_distribution,
    plot_6m_yoy_distribution,
    plot_ind_var_trends,
    )
    
    df = df.copy()

    all_charts = {} # collects all charts across all models
    all_fcst_distributions = {}
    
    # ============================================================
    # MAIN LOOP — one dependent variable at a time
    # ============================================================
    for dep, cfg_model in MODEL_CONFIG.items():

        charts = {} # collects charts per model
        fcst_distributions = {}

        print(f"📊 Creating charts for: {dep}")

        # --------------------------------------------------------
        # Apply chart window settings
        # --------------------------------------------------------
        start_train_chart  = cfg_run.end_train - pd.DateOffset(years=cfg_run.fit_chart_years)
        start_fcst_chart   = cfg_run.end_fcst  - pd.DateOffset(years=cfg_run.fcst_chart_years)
       
        # Training window slice (fit charts)
        df_train_chart = df.loc[start_train_chart:cfg_run.end_train].copy()

        # Observed + forecast window slice
        df_fcst_chart = df.loc[start_fcst_chart:cfg_run.end_fcst].copy()

        # --------------------------------------------------------
        # Forecast level chart
        # --------------------------------------------------------      
        fig = plot_observed_forecast(
            df=df_fcst_chart,
            dep=dep,
            fit=f"{dep}_fit",
            forecast=f"{dep}_forecast",
            start_date_forecast=cfg_run.start_fcst
        )
        charts[f"{dep}_level_forecast"] = fig
        plt.close(fig)

        # YOY forecast
        fig = plot_observed_forecast_yoy(
            df=df_fcst_chart,
            dep=f"{dep}_yoy",
            fit=f"{dep}_fit_yoy",
            forecast=f"{dep}_forecast_yoy",
            start_date_forecast=cfg_run.start_fcst
        )
        charts[f"{dep}_level_forecast_yoy"] = fig
        plt.close(fig)

        # --------------------------------------------------------
        # Independent variable trend charts
        # --------------------------------------------------------
        for var in cfg_model.yoy_charts:
            fig = plot_ind_var_trends(df_fcst_chart, var, cfg_run.start_fcst)
            charts[var] = fig
            plt.close(fig)

        fig = plot_ind_var_trends(df_fcst_chart, f"{dep}_yoy", cfg_run.start_fcst)
        charts[f"{dep}_yoy"] = fig
        plt.close(fig)

        # --------------------------------------------------------
        # Probability distributions (level + YOY)
        # --------------------------------------------------------
        if cfg_model.type == "stat_model":
            sims = upstream_sims_dict.get(dep)
        else:
            sims = downstream_sims_dict.get(dep)

        if sims is None:
            raise ValueError(f"No sims found for dep: {dep}")

        agg = cfg_model.agg_rule
        N_BINS = 30

        # ------------------
        # LEVEL aggregation
        # ------------------
        if agg == "sum":
            sixm_level = sims.sum(axis=1)
        elif agg == "mean":
            sixm_level = sims.mean(axis=1)
        else:
            raise ValueError(f"Unknown agg_rule: {agg}")

        # Plot LEVEL distribution (helpers compute histogram internally)
        fig = plot_6m_distribution(
            sims=sixm_level,
            dep=dep,
            bins=N_BINS
        )
        charts[f"{dep}_forecast_distribution"] = fig
        plt.close(fig)

        # ------------------
        # YOY aggregation
        # ------------------
        ly_start = cfg_run.start_fcst - pd.DateOffset(years=1)
        ly_end   = cfg_run.end_fcst   - pd.DateOffset(years=1)

        if agg == "sum":
            ly_level = df.loc[ly_start:ly_end, dep].sum()
        elif agg == "mean":
            ly_level = df.loc[ly_start:ly_end, dep].mean()

        sims_yoy = (sixm_level - ly_level) / ly_level

        fig = plot_6m_yoy_distribution(
            sims=sims_yoy,
            dep=dep,
            bins=N_BINS
        )
        charts[f"{dep}_forecast_distribution_yoy"] = fig
        plt.close(fig)

        # ------------------------------------------
        # Persist distribution data (BIN-LEVEL)
        # ------------------------------------------
        counts_lvl, edges_lvl = np.histogram(sixm_level, bins=N_BINS)
        probs_lvl = counts_lvl / counts_lvl.sum()
        centers_lvl = (edges_lvl[:-1] + edges_lvl[1:]) / 2

        counts_yoy, edges_yoy = np.histogram(sims_yoy, bins=N_BINS)
        probs_yoy = counts_yoy / counts_yoy.sum()
        centers_yoy = (edges_yoy[:-1] + edges_yoy[1:]) / 2

        dist_df = pd.concat([
            pd.DataFrame({
                "dep": dep,
                "horizon": "6m",
                "metric": "level",
                "bin_id": np.arange(len(centers_lvl)),
                "bin_center": centers_lvl,
                "probability": probs_lvl,
            }),
            pd.DataFrame({
                "dep": dep,
                "horizon": "6m",
                "metric": "yoy",
                "bin_id": np.arange(len(centers_yoy)),
                "bin_center": centers_yoy,
                "probability": probs_yoy,
            }),
        ], ignore_index=True)

        fcst_distributions[dep] = dist_df

        # collect distributions into 
        all_fcst_distributions.update(fcst_distributions)

        # 👇 POUR charts into all_charts
        all_charts.update(charts)

    return {"charts": all_charts, "fcst_distributions": all_fcst_distributions}
