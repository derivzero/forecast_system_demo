import os
import psycopg2
from dotenv import load_dotenv

def get_pg_connection():
    load_dotenv()

    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        dbname=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
    )

def fetch_latest_long(conn, table_name: str):
    """
    Fetches the latest ingested data for each period_date from a raw table.
    """
    sql = f"""
        SELECT
            t.ingest_date,
            t.period_date,
            t.var_name,
            t.var_value
        FROM raw.{table_name} t
        JOIN (
            SELECT
                period_date,
                MAX(ingest_date) AS max_ingest_date
            FROM raw.{table_name}
            GROUP BY period_date
        ) m
          ON t.period_date = m.period_date
         AND t.ingest_date = m.max_ingest_date
        ORDER BY t.period_date, t.var_name
    """
    return pd.read_sql(sql, conn)


def pivot_long_to_wide(df_long):
    """
    Pivots long format dataframe to wide format.
    """
    wide = (
        df_long
        .pivot(index="period_date", columns="var_name", values="var_value")
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


