        
import pandas as pd
import os, sys
from pathlib import Path
from py_files.client_dataset import client_data
from py_files.run_configs import RUN_CONFIG as cfg_run

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)  # For file paths
sys.path.append(PROJECT_ROOT)  # For module imports

def create_external_datasets():
       
    ingest_ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    fred_path = Path(PROJECT_ROOT) / "data" / "shared" / "fetch_fred_data.csv"
    fred_dataset = pd.read_csv(fred_path)
    fred_dataset['date'] = pd.to_datetime(fred_dataset['date'], format="%Y-%m-%d")
    fred_dataset = fred_dataset.set_index("date")
    fred_dataset = fred_dataset.sort_index()
    fred_dataset = fred_dataset.loc[:cfg_run.end_train]
    fred_dataset = fred_dataset.reset_index()
    fred_dataset["date"] = pd.to_datetime(fred_dataset["date"]).dt.date
    #fred_dataset.to_csv("fred.csv")
    
    # fetch forward dataset
    forward_path = Path(PROJECT_ROOT) / "data" / "shared" / f"forward_values_{cfg_run.forecast_month}.csv"
    forward_dataset = pd.read_csv(forward_path)
    forward_dataset['date'] = pd.to_datetime(forward_dataset['date'], format="%m/%d/%Y")
    forward_dataset = forward_dataset.reset_index()
    forward_dataset["date"] = pd.to_datetime(forward_dataset["date"]).dt.date
    #forward_dataset.to_csv("forward.csv")

    # create client dataset
    client_dataset = client_data()
    client_dataset = client_dataset.sort_index()
    client_dataset = client_dataset.loc[:cfg_run.end_train]
    client_dataset = client_dataset.reset_index()
    client_dataset["date"] = pd.to_datetime(client_dataset["date"]).dt.date
    #client_dataset.to_csv("client.csv")

    return fred_dataset, forward_dataset, client_dataset
    
    