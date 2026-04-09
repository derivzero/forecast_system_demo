import pandas as pd
import streamlit as st
import snowflake.connector
from cryptography.hazmat.primitives import serialization

# ----------------------------------------------
# CONNECTION FUNCTIONS
# -----------------------------------------------
def get_sf_connection():
    import os
    from dotenv import load_dotenv
    import snowflake.connector
    from cryptography.hazmat.primitives import serialization

    load_dotenv()

    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    if not key_path:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY_PATH not set")

    with open(key_path, "rb") as key_file:
        p_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
        )

    pkb = p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        private_key=pkb,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )

# ----------------------------------------------
# FETCH FUNCTIONS - SF TO PY
# -----------------------------------------------
def fetch_latest_long(conn, table_name: str):
    sql = f"""
        SELECT
            t.INGEST_DATE,
            t.PERIOD_DATE,
            t.VAR_NAME,
            t.VAR_VALUE
        FROM forecast_db.raw.{table_name} t
        JOIN (
            SELECT
                PERIOD_DATE,
                MAX(INGEST_DATE) AS MAX_INGEST_DATE
            FROM forecast_db.raw.{table_name}
            GROUP BY PERIOD_DATE
        ) m
          ON t.PERIOD_DATE = m.PERIOD_DATE
         AND t.INGEST_DATE = m.MAX_INGEST_DATE
        ORDER BY t.PERIOD_DATE, t.VAR_NAME
    """
    return pd.read_sql(sql, conn)

# -------------------------------------
# Data manip
# -------------------------------------

def pivot_long_to_wide(df_long):
    wide = (
        df_long
        .pivot(index="PERIOD_DATE", columns="VAR_NAME", values="VAR_VALUE")
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide

# changes NaN to None. SF can handle NaN
def df_nan_to_none(df):
    """
    Replace NaN with None by forcing object dtype.
    """
    return df.astype(object).where(df.notna(), None)

