import snowflake.connector
import pandas as pd
import datetime
import streamlit as st
import snowflake.connector

# ----------------------------------------------
# BUILD FUNCTIONS - PY to SF
# -----------------------------------------------

def build_chart_registry(chart_registry_dict, run_id, publish_date):
    """
    Build chart registry table from chart registry dictionary.
    This is metadata only — no charts, no inference.
    """

    rows = []

    for chart_key, spec in chart_registry_dict.items():
        rows.append({
            "run_id": run_id,
            "publish_date": publish_date,
            "chart_key": chart_key,
            "chart_group": spec["chart_group"],
            "chart_type": spec["chart_type"],
            "var_name": spec["var_name"]
        })

    return (
        pd.DataFrame(rows)
          .sort_values(["chart_group", "chart_key"])
          .reset_index(drop=True)
    )


def build_meta_registry(ALL_META, run_id, publish_date):
    """
    Build a meta registry DataFrame from ALL_META.
    """

    rows = []

    for meta_key, meta_value in ALL_META.items():

        # stringify values safely
        if isinstance(meta_value, (dict, list)):
            meta_value_str = json.dumps(meta_value)
        else:
            meta_value_str = str(meta_value)

        rows.append({
            "run_id": run_id,
            "publish_date": publish_date,
            "meta_key": meta_key,
            "meta_value": meta_value_str
        })

    return (
        pd.DataFrame(rows)
          .sort_values("meta_key")
          .reset_index(drop=True)
    )

# breaking up forecast_results wide into three tables
def build_long(df, cols, value_type):
    out = df[["run_id", "period_date", "publish_date"] + cols].melt(
        id_vars=["run_id", "period_date", "publish_date"],
        var_name="series_name",
        value_name="value",
    )
    out["value_type"] = value_type
    return out