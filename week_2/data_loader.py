"""
Week 2: Data Loader Module
Handles downloading, caching, and loading the Telco Customer Churn dataset.
"""

import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def get_data_path() -> str:
    """Returns the primary dataset path, checking local locations first."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            return p

    # Default target path
    return possible_paths[0]


def ensure_dataset() -> str:
    """Ensures the dataset exists locally; downloads it if missing."""
    target_path = get_data_path()
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[+] Using existing dataset at: {target_path}")
        return target_path

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    print(f"[*] Downloading Telco Customer Churn dataset...")

    try:
        urllib.request.urlretrieve(DATA_URL, target_path)
        print(f"[+] Dataset successfully downloaded from primary URL: {target_path}")
    except Exception as e:
        print(f"[!] Primary URL failed ({e}). Trying fallback URL...")
        urllib.request.urlretrieve(FALLBACK_URL, target_path)
        print(f"[+] Dataset successfully downloaded from fallback URL: {target_path}")

    # Also copy to root data/ folder if it exists
    root_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "Telco-Customer-Churn.csv")
    if root_data_path != target_path:
        try:
            os.makedirs(os.path.dirname(root_data_path), exist_ok=True)
            import shutil
            shutil.copy2(target_path, root_data_path)
            print(f"[+] Synced dataset to: {root_data_path}")
        except Exception:
            pass

    return target_path


def load_raw_data() -> pd.DataFrame:
    """Loads the raw customer churn dataset as a pandas DataFrame."""
    path = ensure_dataset()
    df = pd.read_csv(path)
    print(f"[+] Loaded raw dataset with shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head(2))
