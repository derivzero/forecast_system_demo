import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from py_files.run_configs import RUN_CONFIG as cfg_run

# ----------------------------------------------------------
# Optimize Functions
# -----------------------------------------------------------

def create_price_grid_df(df, cfg_units, kf_results_units):

    # guards
    assert cfg_units.name == "units"
    assert "beta_last" in kf_results_units

    # forecast window
    df_fcst = df.loc[cfg_run.start_fcst : cfg_run.end_fcst]

    exp_price         = df_fcst["sales"].sum() / df_fcst["units"].sum()
    exp_price_rounded = round(exp_price, 2)
    prices            = np.round(exp_price_rounded + 0.005 * np.arange(-100, 101), 3)

    beta  = kf_results_units["beta_last"]
    smear = kf_results_units.get("smear", 1.0)

    # base X (forecast window)
    X_base = df_fcst[cfg_units.ind_log].copy()
    X_base = sm.add_constant(X_base, has_constant="add")

    # build grid
    rows = []

    for p in prices:
        Xp = X_base.copy()
        Xp["avg_price_log"] = np.log(p)
        Xp = Xp.reindex(columns=beta.index)

        units = np.exp(Xp @ beta) * smear
        total_units = units.sum()

        rows.append({
            "avg_price": p,
            "units": total_units
        })

    price_grid_df = pd.DataFrame(rows)

    return price_grid_df
      
def tvp_reg_kf_nl(y, X, q, r, beta0=None):
    """
    Minimal TVP-KF runner.
    y : (T,) numpy array
    X : (T, K) numpy array
    q : process noise
    r : observation noise
    beta0 : optional initial state

    Returns dict with:
        - beta_last
        - smear
    """

    import numpy as np

    T, K = X.shape
    beta = np.zeros(K) if beta0 is None else beta0.copy()
    P = np.eye(K) * 1e3
    y_var = np.var(y)

    for t in range(T):
        # prediction
        beta_pred = beta
        P_pred = P + q * np.eye(K)

        # update
        xt = X[t]                     # shape (K,)
        S = xt @ P_pred @ xt + r * y_var
        K_gain = (P_pred @ xt) / S
        err = y[t] - xt @ beta_pred
        beta = beta_pred + K_gain * err
        P = P_pred - np.outer(K_gain, xt) @ P_pred

    # smear correction
    y_hat = X @ beta
    resid = y - y_hat
    smear = np.mean(np.exp(resid))

    return {
        "beta_last": beta,
        "smear": smear
    }

def run_units_nonlinear_kf(df, cfg_units, beta0=None):

    import numpy as np
    import statsmodels.api as sm

    # ---------------------------------------
    # Training window
    # ---------------------------------------
    df_train = df.loc[cfg_run.start_train : cfg_run.end_train]

    # dependent variable (log-units)
    y = np.log(df_train["units"]).values

    # ---------------------------------------
    # Build nonlinear design matrix
    # ---------------------------------------
    X = df_train[cfg_units.ind_log].copy()
    X = sm.add_constant(X, has_constant="add")
    X["avg_price_log_sq"] = X["avg_price_log"] ** 2

    # convert to numpy early
    X = X.values

    # ---------------------------------------
    # Align beta0 to nonlinear X
    # ---------------------------------------
    if beta0 is not None:
        beta0 = np.asarray(beta0).reshape(-1)
        K = X.shape[1]

        if beta0.size == K - 1:
            # linear betas → nonlinear X
            beta0 = np.r_[beta0, 0.0]
        elif beta0.size != K:
            raise ValueError(
                f"beta0 length {beta0.size} does not match X columns {K}"
            )

    # ---------------------------------------
    # Run KF (math-only)
    # ---------------------------------------
    return tvp_reg_kf_nl(
        y=y,
        X=X,
        q=cfg_units.q,
        r=cfg_units.r,
        beta0=beta0
    )

 
