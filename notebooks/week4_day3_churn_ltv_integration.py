"""
Week 4 Day 3 - Churn + LTV Model Integration

This script runs the customer-level integration pipeline and displays
the generated output summary.
"""

from pathlib import Path

import pandas as pd

from src.churn_ltv_integration import build_integration, create_summary


OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "week4_day3_integration"
)

OUTPUT_FILE = OUTPUT_DIR / "churn_ltv_integrated_customer_output.csv"
SUMMARY_FILE = OUTPUT_DIR / "integration_summary.csv"


def main() -> None:
    """Run the integration and display the resulting output."""

    integrated = build_integration()
    summary = create_summary(integrated)

    integrated.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8",
    )

    print("WEEK 4 DAY 3 NOTEBOOK/RUNNER")
    print("=" * 60)
    print("Integrated customers:", len(integrated))
    print("Unique customer IDs:", integrated["customerID"].nunique())
    print("Missing Estimated_LTV:", integrated["Estimated_LTV"].isna().sum())
    print()
    print(summary.to_string(index=False))
    print()
    print("Output:", OUTPUT_FILE)
    print("Summary:", SUMMARY_FILE)
    print("SUCCESS")


if __name__ == "__main__":
    main()