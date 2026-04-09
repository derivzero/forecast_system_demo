import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from py_files.run_configs import RUN_CONFIG as cfg_run
from io import BytesIO

from sklearn.metrics import (mean_squared_error, mean_absolute_error, mean_absolute_percentage_error)

def run_kalman_model(cfg_model, df, df_train_h, df_test_h, df_train, df_future):

    charts = {}
    meta = {}

    """
    1) Runs OLS model to help calculate the R for Kalman
    2) Holdout KF for tuning Q and R
    3) Full Training KF Model. THe last betas are used to forecast forward
    4) Forecast Forward
    """
    from py_files.utils import missing_values
    from py_files.forecast_helpers import (
        tvp_reg_kf, 
        recursive_kf_forecast, 
        ols_sm,
        rolling_6m_mape,
        extract_bootstrap_residuals,
        bootstrap_simulate
    )
    from py_files.charts_helpers import (
        plot_observed_fit_forecast_holdout,
        plot_rolling_mape,
        plot_betas,
        plot_normalized_betas
    )

    # ------------------------------------------------------------
    # 1. OLS for baseline values for Holdout and Full Training Model
    # ------------------------------------------------------------
    ols_df, ols_y_fitted, ols_residuals, ols_model, ols_X = ols_sm(df_train_h, cfg_model.ind_log, cfg_model.dep_log)
    
    resid_var = np.var(ols_residuals)
    y_var = np.var(df_train[cfg_model.dep_log].values)
    r_est = resid_var / y_var
    print(f"Initial R Estimate: {r_est}")

    # Build test matrix
    X_test = sm.add_constant(df_test_h[cfg_model.ind_log], has_constant="add")
    ols_fcst_log = X_test @ ols_model.params

    # Correct smearing estimator
    ols_resid = ols_residuals
    ols_smear = np.mean(np.exp(ols_resid))

    df_test_h['ols_fcst_level'] = np.exp(ols_fcst_log) * ols_smear

    # ------------------------------------------------------------
    # 2. HOLDOUT KF (TUNING VIEW)
    # ------------------------------------------------------------
    
    # Run the Kalman model
    y_fit_log, y_forecast_log, betas_df, output = tvp_reg_kf(
        df_train_h, 
        df_test_h, 
        cfg_model.dep_log, 
        cfg_model.ind_log, 
        cfg_model.q, 
        cfg_model.r
    )

    # Evaluate Holdout Performance
    log_resid = df_train_h[cfg_model.dep_log] - y_fit_log
    smear = np.mean(np.exp(log_resid))
    fcst_level = np.exp(y_forecast_log) * smear # y_forecast_log comes from the model

    # Performance metrics for Holdout
    mse = mean_squared_error(df_test_h[cfg_model.dep], fcst_level)
    mae = mean_absolute_error(df_test_h[cfg_model.dep], fcst_level)
    mape = mean_absolute_percentage_error(df_test_h[cfg_model.dep], fcst_level)

    print("------------------Holdout Performance---------------")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")

    # Create charts
    df_chart = pd.concat([df_train_h, df_test_h]).sort_index()
    df_chart["y_fit_log"] = y_fit_log
    df_chart["y_forecast_log"] = y_forecast_log
    df_chart["y_fit"] = np.exp(y_fit_log) * smear
    df_chart["y_forecast"] = np.exp(y_forecast_log) * smear
    df_chart["ols_forecast"] = df_test_h['ols_fcst_level']

    end_date = df_test_h.index.max()
    start_date = end_date - pd.DateOffset(years=cfg_run.holdout_chart_years)
    df_chart = df_chart.loc[start_date:end_date]

    fig = plot_observed_fit_forecast_holdout(
        df=df_chart, 
        dep=cfg_model.dep,
        ols_dep = 'ols_forecast', 
        fit='y_fit', 
        forecast='y_forecast')
    
    charts[f"{cfg_model.dep}_forecast_holdout"] = fig
    plt.close(fig)

    # Capture the holdout data for SF
    holdout_df = pd.concat(
    [
        df_chart[[cfg_model.dep]]
            .rename(columns={cfg_model.dep: "value"})
            .assign(value_type="actual"),

        df_chart[["y_fit"]]
            .rename(columns={"y_fit": "value"})
            .assign(value_type="fit"),

        df_chart[["y_forecast"]]
            .rename(columns={"y_forecast": "value"})
            .assign(value_type="forecast"),

        df_chart[["ols_forecast"]]
            .rename(columns={"ols_forecast": "value"})
            .assign(value_type="ols"),
    ],
    axis=0,
    )

    # Materialize index → column exactly once
    holdout_df = holdout_df.reset_index().rename(columns={"index": "period_date"})

    # Normalize type for Snowflake
    holdout_df["period_date"] = pd.to_datetime(holdout_df["period_date"]).dt.date

    # Add dep label
    holdout_df["dep"] = cfg_model.dep


    # ------------------------------------------------------------
    # 3. FULL KF TRAINING
    # ------------------------------------------------------------
    
    # Run Kalman model on all historical data so we get the most up-to-date betas for the forecast
    y_fit_log, _, betas_df, output = tvp_reg_kf(
        df_train, 
        None,
        cfg_model.dep_log, 
        cfg_model.ind_log, 
        cfg_model.q, 
        cfg_model.r
    )

    # Forecast inputs from training df
    beta_last = betas_df.iloc[-1]

    kf_results = {
    "beta_last": beta_last,
    "smear": smear
    }

    # --------------------------------------------------------
    # BUILD ONE COMBINED BETA CHART
    # --------------------------------------------------------
    beta_start_chart = cfg_run.end_train - pd.DateOffset(years=cfg_run.beta_chart_years)
    betas_chart = betas_df.loc[beta_start_chart:]
    fig = plot_betas(betas_chart, cfg_model.beta_charts)
    charts[f"{cfg_model.dep}_betas"] = fig
    plt.close(fig)

    # --------------------------------------------------------
    # BUILD ONE COMBINED NORMALIZED BETA CHART
    # --------------------------------------------------------
    fig = plot_normalized_betas(betas_chart, cfg_model.beta_charts)
    charts[f"{cfg_model.dep}_betas_normalized"] = fig
    plt.close(fig)

    # Rolling 6-Month Validation Forecast (Full Train) -------------------
    results_6m, forecasts_6m = rolling_6m_mape(df_train.copy(), cfg_model, cfg_run)

    # extract rolling residuals
    errors_pct = extract_bootstrap_residuals(forecasts_6m, cfg_model.dep)

    # organize residuals by horizon (0-5)
    errors_by_horizon = [errors_pct[:, h] for h in range(6)]

    fig = plot_rolling_mape(results_6m, cfg_model.dep)
    charts[f"{cfg_model.dep}_rolling_6m_mape_holdout"] = fig
    plt.close(fig)

    # ------------------------------------------------------------
    # 4. BUILD BETAS DF
    # ------------------------------------------------------------
    
    # Materialize index → column exactly once
    betas_df = betas_chart.reset_index().rename(columns={"index": "period_date"})

    # Normalize type for Snowflake
    betas_df["period_date"] = pd.to_datetime(betas_df["period_date"]).dt.date

    keep = cfg_model.beta_charts                  # e.g., ["avg_price_log", "cpi_fah", "rdi"]
    betas_df = betas_df[["period_date"] + keep]      # grabs only those columns defned by beta charts in model configs

    # Add dep label
    betas_df["dep"] = cfg_model.dep

    id_cols = ["period_date", "dep"]

    # we want the long format so we melt
    betas_long = (
        betas_df
            .melt(
                id_vars=id_cols,
                var_name="beta_name",
                value_name="beta_value"
            )
    )

    # ------------------------------------------------------------
    # 4. FORECAST
    # ------------------------------------------------------------
    df_future.loc[df_future.index[0], cfg_model.lag_log] = df_train[cfg_model.dep_log].iloc[-1]
    
    fcst_log, df = recursive_kf_forecast(
        df=df,
        df_future=df_future,
        dep_log=cfg_model.dep_log,
        lag_log=cfg_model.lag_log,
        ind_log=cfg_model.ind_log,
        beta_last=beta_last
    )

    # Calculate smear so we can convert from log to level
    if cfg_model.dep_log is not None:
        log_resid = df_train[cfg_model.dep_log].loc[y_fit_log.index] - y_fit_log
        smear = np.mean(np.exp(log_resid))
    else:
        smear = 1.0

    # Convert from log to level on master df. Use slices to insert values based on their index
    fcst_log = df.loc[df_future.index, cfg_model.dep_log].to_numpy(dtype=float)
    fcst_level = np.exp(fcst_log) * smear
    df[f"{cfg_model.dep}_fit"] = np.exp(y_fit_log) * smear # converts the log fit to level fit and adjusts for log bias
    
    df.loc[df_future.index, f"{cfg_model.dep}_forecast"] = fcst_level # applies the forecast level to dep_forecast for just forecast period
    
    df.loc[df_future.index, cfg_model.dep] = df.loc[df_future.index, f"{cfg_model.dep}_forecast"] # appends the forecast level data to the bottom of the actual data
    #df.loc[df_future.index, cfg_model.dep_log] = fcst_log # appends the forecast log data to the bottom of the actual log data
    if cfg_model.lag_log is not None:
        df[cfg_model.lag_log] = df[cfg_model.dep_log].shift(1) # regenerate lagged dependent — now includes forecast values

    # forecast distribution based on residual bootstrap simulations
    errors_pct = extract_bootstrap_residuals(forecasts_6m, cfg_model.dep)
    errors_by_horizon = [errors_pct[:, h] for h in range(6)]
    sim_paths = bootstrap_simulate(fcst_level, errors_by_horizon, n_sim=1000) # error_by_horizon are from the 6 month rolling forecasts

    return {
    "df": df,
    "sim_paths": sim_paths,
    "results_6m": results_6m,
    "kf_results": kf_results,
    "holdout_df": holdout_df,
    "betas_df": betas_long,
    "charts": charts,  
    "meta": {}
    }

