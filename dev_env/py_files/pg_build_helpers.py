import pandas as pd
import json


# ----------------------------------------------
# BUILD FUNCTIONS
# ----------------------------------------------

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


def build_long(df, cols, value_type):
    """
    Break up forecast_results wide into long format.
    """
    out = df[["run_id", "period_date", "publish_date"] + cols].melt(
        id_vars=["run_id", "period_date", "publish_date"],
        var_name="series_name",
        value_name="value",
    )
    out["value_type"] = value_type
    return out