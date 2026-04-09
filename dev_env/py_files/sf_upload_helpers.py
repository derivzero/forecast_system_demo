import pandas as pd
import numpy as np
from io import BytesIO

# ----------------------------------------------
# UPLOAD FUNCTIONS UTILS
# -----------------------------------------------
from io import BytesIO

def fig_to_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    buffer.seek(0)
    image_bytes = buffer.read()
    buffer.close()
    return image_bytes


def publish_charts(run_id, charts, chart_specs, conn):
    sql = """
        INSERT INTO forecast_db.output.chart_artifacts (
            run_id,
            chart_key,
            chart_group,
            chart_type,
            image_data
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    cur = conn.cursor()

    # Guard: prevent republish
    cur.execute("""
        SELECT COUNT(*)
        FROM forecast_db.output.chart_artifacts
        WHERE run_id = %s
    """, (run_id,))

    if cur.fetchone()[0] > 0:
        raise ValueError(f"Run {run_id} already exists. Aborting publish.")

    rows = []

    for chart_key, fig in charts.items():
        image_bytes = fig_to_bytes(fig)
        meta = chart_specs[chart_key]

        rows.append((
            run_id,
            chart_key,
            meta["chart_group"],
            meta["chart_type"],
            image_bytes
        ))

    cur.executemany(sql, rows)
    conn.commit()

# ----------------------------------------------
# UPLOAD FUNCTIONS FOR INPUT DATA
# -----------------------------------------------
def upload_fred_long(conn, fred_wide):
    ingest_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    df = fred_wide.copy()
    df["INGEST_DATE"] = ingest_ts
    df = df.rename(columns={"date": "PERIOD_DATE"})

    long_df = pd.melt(
        df,
        id_vars=["INGEST_DATE", "PERIOD_DATE"],
        var_name="VAR_NAME",
        value_name="VAR_VALUE",
    )

    # normalize types
    long_df["PERIOD_DATE"] = pd.to_datetime(long_df["PERIOD_DATE"]).dt.date

    # Convert NaN -> None so Snowflake inserts NULL
    long_df["VAR_VALUE"] = long_df["VAR_VALUE"].replace({np.nan: None})

    rows = [
        (
            row.INGEST_DATE,
            row.PERIOD_DATE,
            row.VAR_NAME,
            row.VAR_VALUE,
        )
        for row in long_df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO FRED_LONG (
            INGEST_DATE,
            PERIOD_DATE,
            VAR_NAME,
            VAR_VALUE
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()

def upload_forward_long(conn, forward_wide):
    ingest_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    df = forward_wide.copy()
    df["INGEST_DATE"] = ingest_ts
    df = df.rename(columns={"date": "PERIOD_DATE"})

    long_df = pd.melt(
        df,
        id_vars=["INGEST_DATE", "PERIOD_DATE"],
        var_name="VAR_NAME",
        value_name="VAR_VALUE",
    )

    long_df["PERIOD_DATE"] = pd.to_datetime(long_df["PERIOD_DATE"]).dt.date

    # Convert NaN -> None so Snowflake inserts NULL
    long_df["VAR_VALUE"] = long_df["VAR_VALUE"].replace({np.nan: None})

    rows = [
        (
            row.INGEST_DATE,
            row.PERIOD_DATE,
            row.VAR_NAME,
            row.VAR_VALUE,
        )
        for row in long_df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO FORWARD_LONG (
            INGEST_DATE,
            PERIOD_DATE,
            VAR_NAME,
            VAR_VALUE
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()

def upload_client_long(conn, client_wide):
    ingest_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    df = client_wide.copy()
    df["INGEST_DATE"] = ingest_ts
    df = df.rename(columns={"date": "PERIOD_DATE"})

    long_df = pd.melt(
        df,
        id_vars=["INGEST_DATE", "PERIOD_DATE"],
        var_name="VAR_NAME",
        value_name="VAR_VALUE",
    )

    long_df["PERIOD_DATE"] = pd.to_datetime(long_df["PERIOD_DATE"]).dt.date

    # Convert NaN -> None so Snowflake inserts NULL
    long_df["VAR_VALUE"] = long_df["VAR_VALUE"].replace({np.nan: None})

    rows = [
        (
            row.INGEST_DATE,
            row.PERIOD_DATE,
            row.VAR_NAME,
            row.VAR_VALUE,
        )
        for row in long_df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO CLIENT_LONG (
            INGEST_DATE,
            PERIOD_DATE,
            VAR_NAME,
            VAR_VALUE
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()

# --------------------------------------------------------
# Registries Functions
# -------------------------------------------------------
def write_forecast_registry(conn, record: dict):
    sql = """
        INSERT INTO meta.forecast_registry (
            run_id,
            publish_date,
            status,
            code_version,
            code_release_date,
            is_visible,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    params = (
        record["run_id"],
        record["publish_date"],
        record["status"],
        record["code_version"],
        record["code_release_date"],
        record["is_visible"],
        record["notes"],
    )

    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()

def write_chart_registry(conn, df, run_id, publish_date):
    """
    Append chart registry rows into meta.chart_registry.
    Expects df to already contain chart_key, chart_group, chart_template,
    model_name, is_visible, created_at.
    """

    df = df.copy()
    df["run_id"] = run_id
    df["publish_date"] = publish_date

    # NaN -> None
    df = df.where(pd.notna(df), None)

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))

    sql = f"""
        INSERT INTO meta.chart_registry ({col_list})
        VALUES ({placeholders})
    """

    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()


def write_meta_registry(conn, df):
    """
    Append meta registry rows into meta.meta_registry.

    Expected columns in df:
        run_id
        publish_date
        meta_key
        meta_value
    """

    df = df.copy()

    # NaN -> None for Snowflake
    df = df.where(pd.notna(df), None)

    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))

    sql = f"""
        INSERT INTO meta.meta_registry ({col_list})
        VALUES ({placeholders})
    """

    rows = [tuple(row) for row in df.itertuples(index=False, name=None)]

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    cur.close()

# ---------------------------------------------------------------------
# Write Published Data to SF Functions
# ---------------------------------------------------------------------
def write_df_wide(conn, df_wide):

    import pandas as pd
    from py_files.run_configs import DF_COLUMN_REGISTRY

    df = df_wide.copy()

    # Keep only approved columns
    cols_keep = [k for k, v in DF_COLUMN_REGISTRY.items() if v == 1]

    # Insert column order: platform key first
    insert_columns = ["run_id"] + cols_keep

    df = df[insert_columns]

    column_string = ", ".join(insert_columns)
    placeholders = ", ".join(["%s"] * len(insert_columns))

    sql = f"""
        INSERT INTO OUTPUT.DF_WIDE
        ({column_string})
        VALUES ({placeholders})
    """

    data = [tuple(row) for row in df.to_numpy()]

    cur = conn.cursor()
    cur.executemany(sql, data)
    conn.commit()

def write_dial_values_long(conn, dial_values_long):
    """
    Insert holdout results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [
        (
            row.run_id,
            row.metric,
            row.value,
            row.value_type,
        )
        for row in dial_values_long.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.dial_values (
            run_id,
            metric,
            value,
            value_type
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()

def write_tree(conn, df_tree):
    """
    Insert tree results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [
        (
            row.run_id,
            row.var,
            row.ly,
            row.ty,
            row.yoy,
            row.diff,
            row.value_type,
        )
        for row in df_tree.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.trees (
            run_id,
            var,
            ly,
            ty,
            yoy,
            diff,
            value_type
            )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


def write_holdout_results(conn, holdout_df_all):
    """
    Insert holdout results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [
        (
            row.run_id,
            row.dep,
            row.period_date,
            row.value_type,
            row.value,
        )
        for row in holdout_df_all.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.holdout_results (
            run_id,
            dep,
            period_date,
            value_type,
            value
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


def write_mape_results(conn, rolling_mape_all):
    """
    Insert mape results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [
        (
            row.run_id,
            row.period_date,
            row.mape_value,
            row.dep,
        )
        for row in rolling_mape_all.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.mape_results (
            run_id,
            period_date,
            mape_value,
            dep
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


def write_optimization_df_results(conn, price_grid_df, run_id):
    """
    Insert optimization df results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [(run_id, row.avg_price, row.units) for row in price_grid_df.itertuples(index=False)]

    sql = """
        INSERT INTO forecast_db.output.price_optimization_table (run_id, avg_price, units)    
        VALUES (%s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


def write_fcst_distributions_results(conn, fcst_dist_df):
    """
    Insert forecast distribution results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [
        (
            row.run_id,
            row.dep,
            row.horizon,
            row.metric,
            row.bin_id,
            row.bin_center,
            row.probability,
        )
        for row in fcst_dist_df.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.forecast_distributions (
            run_id,
            dep,
            horizon,
            metric,
            bin_id,
            bin_center,
            probability
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)



def write_betas_df_long(conn, betas_df_long):
    """
    Insert betas long df into SF
    """

    rows = [
        (
            row.run_id,
            row.period_date,
            row.dep,
            row.beta_name,
            row.beta_value,
        )
        for row in betas_df_long.itertuples(index=False)
    ]

    sql = """
        INSERT INTO forecast_db.output.betas (
            run_id,
            period_date,
            dep,
            beta_name,
            beta_value
        )
        VALUES (%s, %s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


