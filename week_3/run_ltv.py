"""
Week 3 - Day 4: Run LTV Prediction & Calculation Pipeline
Contributor: Rajarsh

Executes data loading, LTV financial calculation, regression training & evaluation,
generates report plots, and persists model artifacts.
"""

import os
import sys
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week_3.ltv_module import CustomerLTVPipeline


def get_dataset() -> pd.DataFrame:
    """Finds and loads the dataset from available project locations."""
    base_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(base_dir, "data", "Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(base_dir), "week_2", "data", "Telco-Customer-Churn.csv"),
        os.path.join(os.path.dirname(base_dir), "data", "Telco-Customer-Churn.csv"),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            print(f"[+] Loading dataset from: {p}")
            return pd.read_csv(p)

    raise FileNotFoundError("Could not find Telco-Customer-Churn.csv in project data directories.")


def main():
    print("\n" + "#" * 74)
    print("#  WEEK 3 - DAY 4: CUSTOMER LIFETIME VALUE (LTV) ENGINE (RAJARSH)          #")
    print("#  Financial Formulas, ML Regression Modeling, and Customer Tiering        #")
    print("#" * 74 + "\n")

    # 1. Load Data
    df_raw = get_dataset()
    print(f"[+] Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Run Pipeline
    pipeline = CustomerLTVPipeline()
    ltv_df, benchmark_df = pipeline.run_pipeline(df_raw)

    print("\n" + "=" * 74)
    print("  LTV REGRESSION MODEL EVALUATION BENCHMARK")
    print("=" * 74)
    print(benchmark_df.to_string(index=False))

    print("\n" + "=" * 74)
    print("  CUSTOMER VALUE SEGMENTATION SUMMARY (LTV TIERS)")
    print("=" * 74)
    tier_summary = ltv_df.groupby("LTV_Tier", observed=True).agg(
        Customer_Count=("customerID", "count"),
        Mean_Historical_LTV=("Historical_LTV", "mean"),
        Mean_Projected_Future_LTV=("Future_LTV", "mean"),
        Mean_Total_LTV=("Total_LTV", "mean"),
        Total_Revenue_Share=("Total_LTV", lambda x: f"{x.sum() / ltv_df['Total_LTV'].sum():.1%}"),
    ).loc[["Bronze", "Silver", "Gold", "Platinum"]]
    print(tier_summary.to_string())

    print("\n" + "=" * 74)
    print("  LTV CALCULATION SAMPLE (TOP 5 HIGHEST VALUE ACCOUNTS)")
    print("=" * 74)
    top_cols = ["customerID", "tenure", "Contract", "MonthlyCharges", "Historical_LTV", "Future_LTV", "Total_LTV", "LTV_Tier"]
    print(ltv_df.sort_values(by="Total_LTV", ascending=False)[top_cols].head(5).to_string(index=False))
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
