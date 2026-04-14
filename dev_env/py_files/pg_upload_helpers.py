import pandas as pd
import numpy as np
from io import BytesIO
import gc


# ----------------------------------------------
# UPLOAD UTILITY
# ----------------------------------------------

def fig_to_bytes(fig):
    import matplotlib.pyplot as plt
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300)
    buffer.seek(0)
    image_bytes = buffer.read()
    buffer.close()
    plt.close(fig)
    return image_bytes


# ----------------------------------------------
# UPLOAD FUNCTIONS FOR INPUT DATA
# ----------------------------------------------

def upload_long(conn, df_wide, table_name):
    """
    Generic upload function for fred_long, client_long, forward_long.
    Converts wide format to long and inserts into raw schema.
    """
    ingest_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    df = df_wide.copy()
    df["ingest_date"] = ingest_ts
    df = df.rename(columns={"date": "period_date"})

    long_df = pd.melt(
        df,
        id_vars=["ingest_date", "period_date"],
        var_name="var_name",
        value_name="var_value",
    )

    long_df["period_date"] = pd.to_datetime(long_df["period_date"]).dt.date
    long_df["var_value"] = long_df["var_value"].replace({np.nan: None})

    rows = [
        (
            row.ingest_date,
            row.period_date,
            row.var_name,
            row.var_value,
        )
        for row in long_df.itertuples(index=False)
    ]

    sql = f"""
        INSERT INTO raw.{table_name} (
            ingest_date,
            period_date,
            var_name,
            var_value
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()


# ----------------------------------------------
# PUBLISH CHARTS
# ----------------------------------------------

def publish_charts(run_id, charts, chart_specs, conn, batch_size=10):
    sql = """
        INSERT INTO output.chart_artifacts (
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
        FROM output.chart_artifacts
        WHERE run_id = %s
    """, (run_id,))

    if cur.fetchone()[0] > 0:
        raise ValueError(f"Run {run_id} already exists. Aborting publish.")

    chart_items = list(charts.items())

    for i in range(0, len(chart_items), batch_size):
        batch = chart_items[i:i + batch_size]
        rows = []

        for chart_key, fig in batch:
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
        conn.commit()    # ← remove this line
        gc.collect()
        print(f"Published charts {i+1} to {min(i+batch_size, len(chart_items))} of {len(chart_items)}")


# ----------------------------------------------
# REGISTRY FUNCTIONS
# ----------------------------------------------

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


# ----------------------------------------------
# OUTPUT WRITE FUNCTIONS
# ----------------------------------------------

def write_df_wide(conn, df_wide):

    from py_files.run_configs import DF_COLUMN_REGISTRY

    df = df_wide.copy()

    cols_keep = [k for k, v in DF_COLUMN_REGISTRY.items() if v == 1]
    insert_columns = ["run_id"] + cols_keep
    df = df[insert_columns]

    column_string = ", ".join(insert_columns)
    placeholders = ", ".join(["%s"] * len(insert_columns))

    sql = f"""
        INSERT INTO output.df_wide
        ({column_string})
        VALUES ({placeholders})
    """

    data = [tuple(row) for row in df.to_numpy()]

    cur = conn.cursor()
    cur.executemany(sql, data)



def write_dial_values_long(conn, dial_values_long):
    """
    Insert dial values for a single published run.
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
        INSERT INTO output.dial_values (
            run_id,
            metric,
            value,
            value_type
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)


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
        INSERT INTO output.trees (
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
        INSERT INTO output.holdout_results (
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
        INSERT INTO output.mape_results (
            run_id,
            period_date,
            mape_value,
            dep
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)


def write_optimization_df_results(conn, price_grid_df, run_id):
    """
    Insert optimization results for a single published run.
    Append-only. No updates. No deletes.
    """

    rows = [(run_id, row.avg_price, row.units) for row in price_grid_df.itertuples(index=False)]

    sql = """
        INSERT INTO output.price_optimization_table (run_id, avg_price, units)
        VALUES (%s, %s, %s)
    """

    cur = conn.cursor()
    cur.executemany(sql, rows)


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
        INSERT INTO output.forecast_distributions (
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
    Insert betas into PostgreSQL.
    Append-only. No updates. No deletes.
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
        INSERT INTO output.betas (
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