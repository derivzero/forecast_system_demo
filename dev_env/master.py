import io
print("MASTER FILE LOADED FROM:", __file__)
print("io in globals at import time:", "io" in globals())
import os
import pandas as pd
import numpy as np
from io import BytesIO
from pathlib import Path
from datetime import datetime, date, timezone
from pandas.tseries.offsets import DateOffset

from data.api.FRED_API import fetch_fred_data
from data.api.fetch_fred_monthly import refresh_fred_data

from py_files.dataset import create_dataset
from py_files.forecast_core import run_kalman_model
from py_files.forecast_downstream import run_downstream
from py_files.charts_core import run_charts
from py_files.optimize_core import run_optimization
from py_files.utils import build_dial_yoy_block

# PostgreSQL Specific Modules and Functions
from py_files.create_and_upload import create_external_datasets
from py_files.pg_utils_helpers import (get_pg_connection, fetch_latest_long, pivot_long_to_wide, df_nan_to_none)
from py_files.pg_upload_helpers import (upload_long, publish_charts)
from py_files.pg_build_helpers import build_meta_registry
from py_files.pg_upload_helpers import (
    write_forecast_registry,
    write_df_wide,
    write_dial_values_long,
    write_optimization_df_results,
    write_meta_registry,
    write_tree
)

from py_files.pg_chart_specs import CHART_SPECS
from py_files.run_configs import RUN_CONFIG as cfg_run
from py_files.run_configs import DF_COLUMN_REGISTRY
from py_files.model_configs import MODEL_CONFIG


# ============================================================
# BUILD ARTIFACTS FUNCTION
# ============================================================

def build_artifacts():
    """
    Runs the full pipeline and returns in-memory artifacts.
    Safe to call from Streamlit.
    """

    answer = input("Pull fresh data from the FRED API? (y/n): ").strip().lower()

    if answer in ("y", "yes"):
        print("Refreshing FRED data from API...")
        refresh_fred_data()
    else:
        print("Skipping FRED API refresh — using existing data.")
    
    # ----------------------------
    # 0. External datasets
    # ----------------------------
    fred_dataset, forward_dataset, client_dataset = create_external_datasets()

    conn = get_pg_connection()
    upload_long(conn, fred_dataset, "fred_long")
    upload_long(conn, forward_dataset, "forward_long")
    upload_long(conn, client_dataset, "client_long")
    conn.close()

    conn = get_pg_connection()
    fred_long    = fetch_latest_long(conn, "fred_long")
    client_long  = fetch_latest_long(conn, "client_long")
    forward_long = fetch_latest_long(conn, "forward_long")
    conn.close()

    fred_df    = pivot_long_to_wide(fred_long).set_index("period_date")
    client_df  = pivot_long_to_wide(client_long).set_index("period_date")
    forward_df = pivot_long_to_wide(forward_long).set_index("period_date")

    df_wide = client_df.join(fred_df)
    df = create_dataset(df_wide, forward_df)
     
    # ----------------------------
    # 1. Upstream forecasts
    # ----------------------------
    upstream_sims_dict = {}
    ALL_HOLDOUT = []
    ALL_MAPE = []
    ALL_BETAS = []
    ALL_FCST_CHARTS = {}
    ALL_FCST_META = {}

    for name in ["cpi_fah", "units_mkt_trend", "units", "visits", "avg_cost"]:
        cfg_model = MODEL_CONFIG[name]

        fcst_out = run_kalman_model(
            cfg_model,
            df,
            df.loc[cfg_run.start_train_h : cfg_run.end_train_h],
            df.loc[cfg_run.start_test_h  : cfg_run.end_test_h],
            df.loc[cfg_run.start_train   : cfg_run.end_train],
            df.loc[cfg_run.start_fcst    : cfg_run.end_fcst],
        )

        # ✅ Accumulate beta tables, charts, and meta content
        ALL_HOLDOUT.append(fcst_out["holdout_df"]) # holdout is a df so has to be treated differently. It is a list not dict
        ALL_BETAS.append(fcst_out["betas_df"]) # append (lists) and update (dicts) both stack across the forecasts in the loop
        ALL_FCST_CHARTS.update(fcst_out["charts"])
        ALL_FCST_META.update(fcst_out["meta"])
        
        # MAPE Table
        results_6m = fcst_out["results_6m"]
        mape_df = results_6m.copy()
        mape_df = mape_df.rename(columns={"end_date": "period_date", "mape": "mape_value"})
        mape_df["dep"] = cfg_model.dep # this is where the dep gets added
        ALL_MAPE.append(mape_df)
        
        
        df = fcst_out["df"]
        upstream_sims_dict[name] = fcst_out["sim_paths"]

        if name == "units":
            cfg_units = cfg_model
            kf_results_units = fcst_out["kf_results"]
            E_rel = float(kf_results_units["beta_last"]["avg_price_log"])

        if name == "units_mkt_trend":
            E_cat = float(fcst_out["kf_results"]["beta_last"]["avg_price_trend_log"])

    # ---------------------------------------
    # 2. Downstream + optimization + charts
    # ---------------------------------------
    # capture holdout data for holdout charts in publish mode
    holdout_df_long = pd.concat(ALL_HOLDOUT, ignore_index=True)
    
    betas_df_long = pd.concat(ALL_BETAS, ignore_index=True)
            
    rolling_mape_df_long = pd.concat(ALL_MAPE, ignore_index=True) # stacks all mapes by dep
    rolling_mape_df_long = rolling_mape_df_long[["period_date", "mape_value", "dep"]] 
                
    # capture downstream data for preview and publish
    df, downstream_sims_dict, charts_ttm, df_tree = run_downstream(df, df.loc[cfg_run.start_fcst:cfg_run.end_fcst], upstream_sims_dict, MODEL_CONFIG)

    # capture opt data for both preview and publish
    opt_out = run_optimization(df, cfg_units, kf_results_units, {"E_rel": E_rel, "E_cat": E_cat})
    price_grid_df = opt_out["price_grid_df"]
    
    price_grid_df.to_csv("price_grid.csv")

    # capture charts for PY preview mode
    charts_out = run_charts(df, MODEL_CONFIG, upstream_sims_dict, downstream_sims_dict)

    # capture forecast distribution data for PG
    fcst_dist_df_long = pd.concat(charts_out["fcst_distributions"].values(), ignore_index=True)
    
    # dial values -----------------------------------------------------------------------
    dials_base = ["cpi_fah", "oil_prices_lag7", "ppi_farm_products_lag4", "ppi_food_mfg_lag2", "ppi_food_mfg_lag3", "ppi_food_mfg_lag4", "ppi_grocery", 
                  "units_mkt_trend", "sales_mkt_trend", "rdi", "home_price", "sales", "units", "cogs", "gm", "visits", "fixed_cost", "total_cost", "net_income"]
    
    dials_derived = ["avg_price_trend", "avg_price", "avg_cost", "upv"]
    
    ttm_yoy_long = build_dial_yoy_block(
        df=df,
        end_date=cfg_run.end_train,
        months_back=11,
        dials_base=dials_base,
        dials_derived=dials_derived,
        value_type="ttm"
    )

    fcst_yoy_long = build_dial_yoy_block(
        df=df,
        end_date=cfg_run.end_fcst,
        months_back=5,
        dials_base=dials_base,
        dials_derived=dials_derived,
        value_type="forecast"
    )

    dial_values_long = pd.concat([ttm_yoy_long, fcst_yoy_long], axis=0)

    # prep df for publish mode. PG does not have indices
    DF_COLUMNS_KEEP = [k for k, v in DF_COLUMN_REGISTRY.items() if v == 1]
    df = df.reset_index()
    df = df.rename(columns={"index":"period_date"})
    df["period_date"] = pd.to_datetime(df["period_date"]).dt.strftime("%Y-%m-%d")
    df["period_date"] = pd.to_datetime(df["period_date"])
    df = df_nan_to_none(df) # converts NaNs to None, which PG can handle
    df_wide = df[DF_COLUMNS_KEEP]

    # ----------------------------
    # 3. Collect artifacts (non time series)
    # ----------------------------
    ALL_CHARTS = {}
    ALL_META = {}

    # from forecast core 
    ALL_CHARTS.update(ALL_FCST_CHARTS)
    ALL_META.update(ALL_FCST_META)

    # from downstream
    ALL_CHARTS.update(charts_ttm)
    
    # from optimization
    ALL_CHARTS.update(opt_out["charts"])
    ALL_META.update(opt_out["meta"])
    
    # from charts
    ALL_CHARTS.update(charts_out["charts"])
    
    return df_wide, dial_values_long, price_grid_df, df_tree, ALL_CHARTS, ALL_META

# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Preview and forecast validation 
    # --------------------------------------------------------

    df_wide, dial_values_long, price_grid_df, df_tree, ALL_CHARTS, ALL_META  = build_artifacts()

    # META Content ------------------------------------------
    print("\n=== ALL_META INVENTORY ===")
    print("Total meta:", len(ALL_META))

    for i, (k, v) in enumerate(ALL_META.items(), start=1):
        print(f"{i:03d} | {k:40s} | {type(v)}")

    print("\n=== TYPE COUNTS ===")
    type_counts = {}
    for v in ALL_META.values():
        t = str(type(v))
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"{c:3d}  {t}")

    
    # ------------------------------------------------------------
    # Publish to PostgreSQL (explicit, manual)
    # ------------------------------------------------------------
    typed = input("Type True to publish: ")
    PUBLISH = (typed == "True")

    if PUBLISH:
        publish_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        registry_record = {
            "run_id": cfg_run.run_id,
            "publish_date": publish_ts,
            "status": "published",
            "code_version": "V0",
            "code_release_date": date(2026, 1, 15),
            "is_visible": True,
            "notes": "Baseline Forecast",
        }

        conn = get_pg_connection()
        cur = conn.cursor()

        # --------------------------------------------------
        # Guard: prevent duplicate run_id
        # --------------------------------------------------
        cur.execute(
            "SELECT COUNT(*) FROM meta.forecast_registry WHERE run_id = %s",
            (registry_record["run_id"],)
        )
        if cur.fetchone()[0] > 0:
            conn.close()
            raise RuntimeError(
                f"Publish aborted: run_id already exists ({registry_record['run_id']})"
            )

        try:
            # upload rows to meta and registries ------------------------------------
            print(f"Publishing row to meta")
            write_forecast_registry(conn, registry_record)

            print(f"Publishing meta registry")
            meta_registry_df = build_meta_registry(ALL_META, registry_record["run_id"], registry_record["publish_date"])
            write_meta_registry(conn, meta_registry_df)

            print(f"Publishing charts: {len(ALL_CHARTS)} images")
            publish_charts(run_id=registry_record["run_id"], charts=ALL_CHARTS, chart_specs=CHART_SPECS, conn=conn)

            # Publish df wide -----------------------------------
            print(f"Publishing df wide:{len(df_wide,)} rows")
            df_wide["period_date"] = pd.to_datetime(df_wide["period_date"]).dt.date
            df_wide["run_id"] = registry_record["run_id"]
            df_wide = df_nan_to_none(df_wide)
            write_df_wide(conn, df_wide)

            # Publish dial values -----------------------------------
            print(f"Publishing dial values:{len(dial_values_long,)} rows")
            dial_values_long["run_id"] = registry_record["run_id"]
            dial_values_long = df_nan_to_none(dial_values_long)
            write_dial_values_long(conn, dial_values_long)

            # upload price grid -----------------------------------
            print(f"Publishing price_grid_df:{len(price_grid_df)} rows")
            write_optimization_df_results(conn=conn, price_grid_df=price_grid_df, run_id=registry_record["run_id"])

            # upload tree -----------------------------------
            print(f"Publishing tree values:{len(df_tree,)} rows")
            df_tree["run_id"] = registry_record["run_id"]
            write_tree(conn, df_tree)

            # --------------------------------------------------
            # Commit everything at once
            # --------------------------------------------------
            conn.commit()
            print("Publish complete — all data committed successfully.")

        except Exception as e:
            conn.rollback()
            print(f"Publish failed — all changes rolled back. Error: {e}")
            raise e

        finally:
            conn.close()