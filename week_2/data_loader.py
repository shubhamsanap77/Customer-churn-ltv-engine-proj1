"""
Week 2: Data Loader Module
Loads and verifies the Telco Customer Churn dataset.
"""

import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def get_data_path() -> str:
    """Returns local path to the Telco Customer Churn dataset."""
    base_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(base_dir, "data", "Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(base_dir), "data", "Telco-Customer-Churn.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p
    return possible_paths[0]


def load_raw_data() -> pd.DataFrame:
    """Loads the dataset, downloading if not already present."""
    data_path = get_data_path()
    if not os.path.exists(data_path) or os.path.getsize(data_path) < 1000:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"[*] Downloading Telco Customer Churn dataset...")
        urllib.request.urlretrieve(DATA_URL, data_path)

    df = pd.read_csv(data_path)
    print(f"[+] Loaded raw dataset successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head(2))
