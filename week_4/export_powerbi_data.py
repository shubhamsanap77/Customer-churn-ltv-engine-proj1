"""
Week 4 - Day 4: Power BI Data Integration & Export Pipeline
Contributor: Rajesh (Rajarsh)

Takes raw customer data, applies Week 2 Churn scoring (probabilities & risk categories),
applies Week 3 LTV calculation & regression (Historical, Future, Total LTV, Tiers),
and maps the Strategic 2x2 Retention Matrix (VIP at Risk, Loyal Champions, etc.).
Exports full denormalized and star-schema datasets ready for Power BI Desktop.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd

# Add repo root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_DIR)

from week_2.feature_engineering import FeatureEngineer
from week_3.ltv_module import LTVCalculator, LTVRegressor


def run_powerbi_data_pipeline():
    print("\n" + "=" * 74)
    print("  WEEK 4 - DAY 4: POWER BI DATA INTEGRATION & EXPORT PIPELINE")
    print("=" * 74)

    # 1. Load Data
    data_path = os.path.join(REPO_DIR, "week_2", "data", "Telco-Customer-Churn.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(REPO_DIR, "data", "Telco-Customer-Churn.csv")
    df_raw = pd.read_csv(data_path)
    print(f"[+] Loaded raw dataset: {df_raw.shape[0]} customers, {df_raw.shape[1]} columns")

    # 2. Clean numerical values
    df_clean = df_raw.copy()
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"] * df_clean["tenure"]).round(2)

    # 3. Apply Week 2 Churn Scoring (Logistic Regression)
    print("[*] Generating Churn Probabilities & Risk Levels (Week 2 Model)...")
    churn_model_path = os.path.join(REPO_DIR, "week_2", "models", "logistic_regression_model.pkl")
    churn_scaler_path = os.path.join(REPO_DIR, "week_2", "models", "scaler.pkl")

    fe = FeatureEngineer()
    featured_df = fe.engineer_features(df_raw)
    feature_cols = [c for c in fe.engineer_features(df_raw).columns if c != "Churn"]

    with open(churn_model_path, "rb") as f:
        churn_model = pickle.load(f)
    with open(churn_scaler_path, "rb") as f:
        churn_scaler = pickle.load(f)

    # Align features exactly with model
    with open(os.path.join(REPO_DIR, "week_2", "models", "model_metadata.json"), "r") as f:
        meta_churn = json.load(f)
    expected_churn_features = meta_churn["feature_names"]

    X_churn = featured_df.drop(columns=["Churn"], errors="ignore")
    # Ensure all expected columns exist
    for col in expected_churn_features:
        if col not in X_churn.columns:
            X_churn[col] = 0
    X_churn = X_churn[expected_churn_features]

    # Continuous scaling
    num_cols_churn = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "monthly_to_total_ratio", "avg_historical_monthly", "bill_shock", "service_bundle_count"] if c in X_churn.columns]
    X_churn_scaled = X_churn.copy()
    X_churn_scaled[num_cols_churn] = churn_scaler.transform(X_churn[num_cols_churn])

    churn_probs = churn_model.predict_proba(X_churn_scaled)[:, 1]
    df_clean["Churn_Probability"] = np.round(churn_probs, 4)
    df_clean["Churn_Probability_Pct"] = np.round(churn_probs * 100, 2)
    df_clean["Predicted_Churn"] = np.where(churn_probs >= 0.50, "Yes", "No")

    # Risk Categories
    df_clean["Churn_Risk_Level"] = pd.cut(
        df_clean["Churn_Probability"],
        bins=[-0.01, 0.35, 0.60, 1.01],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    ).astype(str)

    # 4. Apply Week 3 LTV Calculations & Segmentations
    print("[*] Calculating Historical, Future, and Total LTV (Week 3 Engine)...")
    calc = LTVCalculator(gross_margin=0.75)
    ltv_df = calc.calculate_total_ltv(df_clean)

    df_clean["Historical_LTV"] = ltv_df["Historical_LTV"]
    df_clean["Projected_Future_LTV"] = ltv_df["Future_LTV"]
    df_clean["Total_Expected_LTV"] = ltv_df["Total_LTV"]
    df_clean["LTV_Tier"] = ltv_df["LTV_Tier"]

    # 5. Strategic 2x2 Retention Quadrant Matrix
    print("[*] Mapping 2x2 Strategic Retention Matrix...")
    def assign_retention_quadrant(row):
        is_high_value = row["LTV_Tier"] in ["Gold", "Platinum"]
        is_high_risk = row["Churn_Risk_Level"] == "High Risk"

        if is_high_value and is_high_risk:
            return "VIP at Risk"
        elif is_high_value and not is_high_risk:
            return "Loyal Champions"
        elif not is_high_value and is_high_risk:
            return "Low-Value Churn"
        else:
            return "Growth Opportunities"

    def assign_recommended_action(quadrant):
        actions = {
            "VIP at Risk": "High-Touch Rescue: Assign dedicated account manager, offer 20% annual contract discount or free speed upgrade.",
            "Loyal Champions": "Retention & Loyalty: Provide anniversary rewards, priority customer support, and multi-service bundles.",
            "Low-Value Churn": "Automated Re-engagement: Deploy automated digital email survey, offer self-service $10 billing credits.",
            "Growth Opportunities": "Nurture & Upsell: Educational content on streaming/security add-ons, promote 1-year contract migration.",
        }
        return actions.get(quadrant, "Standard monitoring")

    df_clean["Retention_Segment"] = df_clean.apply(assign_retention_quadrant, axis=1)
    df_clean["Recommended_Action"] = df_clean["Retention_Segment"].apply(assign_recommended_action)

    # Summary of Retention Quadrants
    print("\n[+] Retention Quadrants Distribution:")
    quadrant_summary = df_clean.groupby("Retention_Segment").agg(
        Customers=("customerID", "count"),
        Avg_Churn_Prob=("Churn_Probability", "mean"),
        Avg_Total_LTV=("Total_Expected_LTV", "mean"),
        Total_Revenue_At_Stake=("Total_Expected_LTV", "sum")
    ).reset_index()
    for _, q in quadrant_summary.iterrows():
        print(f"    - {q['Retention_Segment']:<22}: {q['Customers']:>4} accounts | Avg LTV: ${q['Avg_Total_LTV']:>8,.2f} | Total: ${q['Total_Revenue_At_Stake']:>11,.2f}")

    # 6. Export Datasets for Power BI
    output_dirs = [
        os.path.join(REPO_DIR, "week_4", "data"),
        os.path.join(REPO_DIR, "dashboard")
    ]
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)

    for out_dir in output_dirs:
        # 1. Master Denormalized Table
        master_path = os.path.join(out_dir, "powerbi_customer_churn_ltv_master.csv")
        df_clean.to_csv(master_path, index=False)
        print(f"[+] Saved Master Power BI Dataset: {master_path}")

        # 2. Star Schema Tables
        dim_cust = df_clean[["customerID", "gender", "SeniorCitizen", "Partner", "Dependents"]].copy()
        dim_cust.to_csv(os.path.join(out_dir, "dim_customers.csv"), index=False)

        dim_contract = df_clean[["customerID", "Contract", "PaperlessBilling", "PaymentMethod"]].copy()
        dim_contract.to_csv(os.path.join(out_dir, "dim_contracts.csv"), index=False)

        service_cols = [
            "customerID", "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ]
        dim_services = df_clean[service_cols].copy()
        dim_services.to_csv(os.path.join(out_dir, "dim_services.csv"), index=False)

        fact_cols = [
            "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn",
            "Churn_Probability", "Churn_Probability_Pct", "Predicted_Churn", "Churn_Risk_Level",
            "Historical_LTV", "Projected_Future_LTV", "Total_Expected_LTV", "LTV_Tier",
            "Retention_Segment", "Recommended_Action"
        ]
        fact_table = df_clean[fact_cols].copy()
        fact_table.to_csv(os.path.join(out_dir, "fact_churn_ltv_predictions.csv"), index=False)

    print("\n[+] All Power BI datasets generated and exported successfully!")
    return df_clean


if __name__ == "__main__":
    run_powerbi_data_pipeline()
