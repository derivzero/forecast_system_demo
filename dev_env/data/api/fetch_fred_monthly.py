# THIS IS THE FILE THAT SHOULD BE EXECUTED TO UPDATE THE ECON DATA

import datetime
from data.api.FRED_API import fetch_fred_data
import os
import pandas as pd


def refresh_fred_data():
    # base_dir returns the api folder which is where fetch_fred lives
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))

    # define the api key path
    api_key_path = os.path.join(PROJECT_ROOT, "data", "api", "fred_api_key.txt")

    # output_path is data shared so we move up one level
    output_path = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "shared", 'fetch_fred_data.csv'))

    # Set full range. End data pulls todays date to catch the most uptodate data
    start_date = "2000-01-01"
    end_date = datetime.date.today().strftime("%Y-%m-%d")

    # Call the API function (fetch_fred_data in FRED_APY.py) and passes in the date range from above and then saves the df to csv
    df = fetch_fred_data(start_date, end_date, api_key_path)

    # Save the CSVs to the correct folders
    df.to_csv(output_path, index=True)

    print(f"✅ Data refreshed and saved on {datetime.datetime.now().isoformat()}")

    return df

if __name__ == "__main__":
    refresh_fred_data()

