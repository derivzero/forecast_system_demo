import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter as SMKalmanFilter

# --------------------------------------------------
# TVP REGRESSION: KALMAN FILTER WITH DRIFTING BETAS
# --------------------------------------------------
    
# Kalman Model used for Training
def tvp_reg_kf(df_train, df_test, dep_log, ind_log, q, r):
    """
    Time-varying-parameter regression via Kalman Filter.
    Training uses prediction + update.
    Forecast uses prediction only (betas frozen at last training estimate).
    """

    # this allows us to run on just training data.  Without this it would error if there was no df_test.
    if df_test is None:
        df_test = df_train.iloc[0:0]

    # ---------------------------
    # 1. Build training matrices
    # ---------------------------
    X_train = sm.add_constant(df_train[ind_log]) # This is a matrix. Columns are regressors and rows are training periods
    y_train = df_train[dep_log].values.astype(float) # values converts series to an array 
    X_test  = sm.add_constant(df_test[ind_log])

    m = X_train.shape[1]     # number of regression coefficients. The columns in the Training matrix. The regressors are the states that move over time
    T = X_train.shape[0]     # number of training periods.  The rows in the Training matrix

    # -------------------------------------
    # 2. Priors for betas and covariance (from OLS for Units only)
    # -------------------------------------
    beta0 = np.zeros(m)              # Starting guess. Initializes prior values which we assume no information so betas all start at 0. m = number of betas
    P0 = np.eye(m) * 1e3             # Confidence in the starting guess.  1e3 suggests Low confidence

    # -------------------------------
    # 3. Build Kalman filter object. This is just defining the model.  Nothing is run at this point
    # -------------------------------
    kf = SMKalmanFilter(k_endog=1, k_states=m) # Just a shell. 1 means one value per period adn m is one state per regression coef including intercept
    kf.bind(y_train)                           # binds observed values to kf. Lets the model know how many periods based on y
    kf.initialize_known(beta0, P0)             # Set initial values. Tells the kf model where to start

    # DESIGN MATRIX: shape (1, m, T)
    kf["design"] = X_train.values.T.reshape(1, m, T) # for y, estimate m betas for each period. Values creates array and then transposed (m, T). Reshape add the y array so we get (1, m, T). The 1 dimension is empty and just inserted so matrix mult can happen downstream
    
    # STATE EVOLUTION: random walk on betas.  How the betas move over time.  If nothing pushes them, they stay the same
    kf["transition"] = np.eye(m)

    # SELECTION: identity → each beta has independent noise.  One beta will not impact another.  Each beta drifts independently
    kf["selection"]  = np.eye(m)

    # STATE COVARIANCE (Q): how much do we want beta's to drift or how fast we want then to change
    kf["state_cov"]  = np.eye(m) * q

    # OBSERVATION NOISE (R): How confident are we in the data.  Small r, we trust the data, big r we do not
    kf["obs_cov", 0, 0] = np.var(y_train) * r

    # ---------------------------
    # 4. Run the Kalman filter
    # ---------------------------
    output = kf.filter() # runs the kf forward through time. For each step two things happen 1) initial prediction 2) updated estimate.  Kalman gain is calculated looped through each period
    # output is a (m,T) matrix of betas

    # FILTERED BETAS: array shape (T, m) - transpose to get periods back as rows and betas as columns for betas_df
    beta_filt = output.filtered_state.T

    # Store betas in DataFrame
    betas_df = pd.DataFrame(
        beta_filt,
        index=df_train.index,
        columns=X_train.columns
    )

    # --------------------------------
    # 5. Use the time varying betas to calculate y
    # --------------------------------
    y_fit = (X_train.values * beta_filt).sum(axis=1) # this is just linear multiplication one row at a time
    y_fit = pd.Series(y_fit.astype(float), index=df_train.index)

    # -------------------------------
    # 6. Forecast using last betas
    # -------------------------------
    beta_last = beta_filt[-1]
    y_test = X_test.values @ beta_last # this is matrix multiplication because we only have one set of betas. X_test (T,m) and beta_last (m,)
    y_test = pd.Series(y_test.astype(float), index=df_test.index)

    return y_fit, y_test, betas_df, output

# Uses the latest betas to forecast forward
def recursive_kf_forecast(
    df,                      # full df (historical + forward rows)
    df_future,               # future slice of df
    dep_log,                 # 'cpi_fah_log'
    lag_log,                 # 'cpi_fah_lag1_log'
    ind_log,                     # list of independent vars including lag
    beta_last                # last filtered betas
):
    """
    Recursively forecast a Kalman regression with a lagged dependent.
    """
    fcst_log = []   # store log forecasts

    # Just confirms shapes align
    if len(ind_log) + 1 != len(beta_last):
        raise ValueError(
            f"recursive_kf_forecast mismatch: "
            f"len(ind_log)+1={len(ind_log)+1}, "
            f"len(beta_last)={len(beta_last)}. "
            f"ind_log={ind_log}"
    )
    
    # -------------------------
    # CASE 1A: NO LAG VARIABLE
    # -------------------------
    if lag_log is None:
        
        # just use X_future @ beta_last, no recursion needed if no lagged dependent
        X_future = df_future[ind_log].values.astype(float)
        fcst_log = list(X_future @ beta_last)

        # store log-forecasts in df
        df.loc[df_future.index, dep_log] = fcst_log
        return fcst_log, df
    
    # -------------------------------
    # CASE 1B: LAG VARIABLE
    # -------------------------------
    x0 = df_future.iloc[0][ind_log].values.astype(float) # the seed. It supplies the first value in the test period

    # prepend intercept
    x0_const = np.concatenate([[1.0], x0]) # Combines the const and X_future values for the first period ONLY. Inserted into loop below for recursion
    log_y0 = float(x0_const @ beta_last) # multiplies the X_future values by betas to estimate y per future period
    fcst_log.append(log_y0) # appends forecast back to fcst_log

    # Write forecast for the first future period. All part of seeding that first forecast update
    df.loc[df_future.index[0], dep_log] = log_y0

    # Move this forecast into the lag column for the next period (P2)
    if len(df_future) > 1:
        df_future.loc[df_future.index[1], lag_log] = log_y0
        df.loc[df_future.index[1], lag_log] = log_y0    

    # -------------------------------
    # RECURSIVE FORECAST LOOP: t = 0 is forecasted above and used to forecast t = 1 (second row) in the loop below
    # -------------------------------
    for t in range(1, len(df_future)):
        
        xt = df_future.iloc[t][ind_log].values.astype(float) # X values, not including constant
        xt_const = np.concatenate([[1.0], xt])   # combine intercept and X values for period t
        log_yt = float(xt_const @ beta_last)     # multiply values and betas for period t

        # NEW: write the forecast for this period into df
        df.loc[df_future.index[t], dep_log] = log_yt

        # This is the recursive part for lag dep.  Feed forecast log_yt above into the next period's lag
        if (t + 1) < len(df_future):
            df_future.loc[df_future.index[t+1], lag_log] = log_yt
            df.loc[df_future.index[t+1], lag_log] = log_yt

    return fcst_log, df


# Standard Statsmodel OLS
def ols_sm(df, ind_log, dep_log):
    """
    Runs an OLS regression using statsmodels and adds fitted values and residuals to the DataFrame.

    Parameters:
    - df: pd.DataFrame
    - ind: list of str, independent variable names
    - dep: str, dependent variable name

    Returns:
    - y_fitted: pd.Series of predicted values
    - residuals: pd.Series of residuals
    - summary: regression summary as text
    - model: the statsmodels regression object
    """

    X = df[ind_log]
    y = df[dep_log]

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    y_fitted = model.fittedvalues
    residuals = model.resid
    
    # Predictions
    predictions = model.get_prediction(X)
    pred_summary = predictions.summary_frame(alpha=0.05)

    # Print regression summary
    print(model.summary())

    # Add fitted values and residuals to the original DataFrame
    df = df.copy() # create copy so it will return a df with y_fitted and residuals
    df.loc[:, 'y_fitted'] = y_fitted
    df.loc[:, 'residuals'] = residuals
    df[f'{dep_log}_se_mean'] = pred_summary['mean_se']
    df[f'{dep_log}_se_lower'] = pred_summary['mean_ci_lower']
    df[f'{dep_log}_se_upper'] = pred_summary['mean_ci_upper']
    df[f'{dep_log}_obs_lower'] = pred_summary['obs_ci_lower']
    df[f'{dep_log}_obs_upper'] = pred_summary['obs_ci_upper']
    df[f'{dep_log}_se_obs'] = (df[f'{dep_log}_obs_upper'] - df[f'{dep_log}_obs_lower']) / (2 * 1.96)

    # this returns the updated df with y_fitted and residuals, the model summary and the model
    return df, y_fitted, residuals, model, X

# Based on the rolling 6 month forecast, it calculates the MAPE over time
def rolling_6m_mape(df, cfg_model, cfg_run):
    """
    Rolling 6-month MAPE evaluation in **LEVEL space**.
    """

    # -------------------------------------
    # 1. Slice training + holdout windows
    # -------------------------------------
    start_train_h = pd.to_datetime(cfg_run.start_train_h)
    end_train_h   = pd.to_datetime(cfg_run.end_train_h)

    start_test_h = pd.to_datetime(cfg_run.start_test_h)
    end_test_h   = pd.to_datetime(cfg_run.end_test_h)

    df_train_h = df.loc[start_train_h : end_train_h].copy()
    df_test_h  = df.loc[start_test_h  : end_test_h ].copy()

    holdout_size = len(df_test_h)     # typically 24 months
    horizon = 6                       # 6-month forecasting window

    results_6m = []
    forecasts_6m = []

    # -------------------------------------
    # 2. Rolling forecast windows
    # -------------------------------------
    for origin in range(0, holdout_size - horizon + 1):

        # expand training window by sliding one more month each iteration
        df_train_window = pd.concat([
            df_train_h,
            df_test_h.iloc[:origin]
        ])

        df_test_window = df_test_h.iloc[origin : origin + horizon]

        # -------------------------------------
        # 2B. Kalman forecast in LOG SPACE
        # -------------------------------------
        _, y_fcst_log, _, _ = tvp_reg_kf(
            df_train=df_train_window,
            df_test=df_test_window,
            dep_log=cfg_model.dep_log,
            ind_log=cfg_model.ind_log,
            q=cfg_model.q,
            r=cfg_model.r
        )

        # -------------------------------------
        # 3. Convert to LEVEL
        # -------------------------------------
        y_pred_level = np.exp(y_fcst_log.values)

        # actual LEVEL series
        actual_level = df_test_window[cfg_model.dep].values

        # -------------------------------------
        # 4. Compute LEVEL MAPE
        # -------------------------------------
        mape_level = np.mean(np.abs((y_pred_level - actual_level) / actual_level)) * 100

        # save evaluation stats
        results_6m.append({
            "origin": origin,
            "start_date": df_test_window.index[0],
            "end_date": df_test_window.index[-1],
            "mape": mape_level
        })

        # save forecast path for bootstrap simulations
        forecasts_6m.append((df_test_window, y_pred_level))

    # convert results to DataFrame
    results_6m = pd.DataFrame(results_6m)

    return results_6m, forecasts_6m


# used with rolling_6m_mape above. That function outputs forecasts_6m
def extract_bootstrap_residuals(forecasts_6m, dep):
    """
    Convert forecasts_6m into a (19 × 6) matrix of signed percent errors.
    """

    n_runs = len(forecasts_6m)     # should be 19
    horizon = 6

    errors_pct = np.zeros((n_runs, horizon))

    for i, (df_test_window, y_pred_level) in enumerate(forecasts_6m):
        actual_level = df_test_window[dep].values

        # signed percentage residual: (pred - actual) / actual
        err_pct = (y_pred_level - actual_level) / actual_level

        errors_pct[i, :] = err_pct

    return errors_pct

# Based on the residuals in the six month rolling HOLDOUT forecast, the sim estimates the forecast 
def bootstrap_simulate(y_fcst_level, errors_by_horizon, n_sim=1000):
    """
    y_fcst_level      : array of length 6 (Kalman forward forecast in LEVELS)
    errors_by_horizon : list of 6 arrays, each with 19 residuals
    """

    horizon = 6
    sim_paths = np.zeros((n_sim, horizon))

    for s in range(n_sim):
        for h in range(horizon):
            sampled_error = np.random.choice(errors_by_horizon[h])
            sim_paths[s, h] = y_fcst_level[h] * (1 + sampled_error)

    # sim_paths is a dictionary of shape (n_sim, number of forecast months). We get a sim forecast for each month.
    return sim_paths


# Plots each individual 6 month forecast
def rolling_forecast(df, dep_yoy, df_fcst_chart, start_date, end_date, forecast_month):

    # Define the index for the new forecast
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Normalize both indexes to remove time component
    new_fcst_mo = pd.date_range(start, end).normalize()
    df.index = pd.to_datetime(df.index).normalize()
    df_fcst_chart.index = pd.to_datetime(df_fcst_chart.index).normalize()

    # remove current forecast column if it exists
    if forecast_month in df_fcst_chart.columns:
        df_fcst_chart = df_fcst_chart.drop(columns=forecast_month)

    # Build the new fcst table
    new_fcst = pd.DataFrame(index=new_fcst_mo)
    new_fcst = pd.merge(new_fcst, df[dep_yoy], left_index=True, right_index=True, how='inner')
    new_fcst = new_fcst.rename(columns={dep_yoy: forecast_month} )

    # Merge forecast table with new forecast
    rolling_forecast_df = pd.merge(df_fcst_chart, new_fcst, left_index=True, right_index=True, how='outer')
    col = rolling_forecast_df.pop("Actuals")
    rolling_forecast_df.insert(0, "Actuals", col)

    return rolling_forecast_df

if __name__ == "__main__":
    print("This module is not intended to be run directly.")



