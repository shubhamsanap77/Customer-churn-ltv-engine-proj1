"""
Week 4 - Day 4: Power BI Pipeline Orchestrator
Contributor: Rajesh (Rajarsh)

Executes data integration, runs ML inference across the 7,043 customer accounts,
generates denormalized and star-schema datasets for Power BI, and validates DAX assets.
"""

import os
import sys
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week_4.export_powerbi_data import run_powerbi_data_pipeline


def main():
    print("\n" + "#" * 74)
    print("#  WEEK 4 - DAY 4: POWER BI DASHBOARD DEVELOPMENT & INTEGRATION (RAJESH)   #")
    print("#  ML Inference Integration, Strategic Retention Matrix & Power BI Assets #")
    print("#" * 74 + "\n")

    # 1. Run Data Integration & Scoring
    df_scored = run_powerbi_data_pipeline()

    # 2. Executive Metrics Summary
    total_cust = len(df_scored)
    high_risk_cust = (df_scored["Churn_Risk_Level"] == "High Risk").sum()
    total_ltv = df_scored["Total_Expected_LTV"].sum()
    vip_at_risk_rev = df_scored[df_scored["Retention_Segment"] == "VIP at Risk"]["Total_Expected_LTV"].sum()
    vip_at_risk_count = (df_scored["Retention_Segment"] == "VIP at Risk").sum()

    print("\n" + "=" * 74)
    print("  EXECUTIVE PORTFOLIO SUMMARY (POWER BI REPORT DATA)")
    print("=" * 74)
    print(f"  - Total Accounts Analyzed:          {total_cust:,}")
  
    print(f"  - Total Portfolio Expected Value:   ${total_ltv:,.2f}")
    print(f"  - Predicted High Churn Risk:        {high_risk_cust:,} accounts ({high_risk_cust / total_cust:.1%})")
    print(f"  - [ALERT] VIP Accounts at Risk:     {vip_at_risk_count:,} accounts")
    print(f"  - [ALERT] VIP Revenue at Stake:     ${vip_at_risk_rev:,.2f}")
    print(f"  - Potential Rescued Value (50%):    ${vip_at_risk_rev * 0.50:,.2f}")

    print("\n" + "=" * 74)
    print("  POWER BI ASSETS READY FOR DEPLOYMENT")
    print("=" * 74)
    print("  1. Power BI 1-Click Connector:  dashboard/Customer_Churn_LTV.pbids")
    print("  2. Master Dataset:              dashboard/powerbi_customer_churn_ltv_master.csv")
    print("  3. DAX Formulas Library:        dashboard/DAX_MEASURES.dax")
    print("  4. Power Query ETL Script:      dashboard/PowerQuery_ETL.m")
    print("  5. Interactive HTML Preview:    dashboard/index.html")
    print("  6. Step-by-Step Guide:          week_4/POWER_BI_DASHBOARD_GUIDE.md")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
