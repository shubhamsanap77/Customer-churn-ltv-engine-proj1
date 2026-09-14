from .preprocessing import run_preprocessing
from .baseline_report import generate_report


def main():
    print("=" * 60)
    print("CUSTOMER CHURN — DAY 6-7 PIPELINE")
    print("=" * 60)

    result = run_preprocessing()

    print("\nPreprocessing complete.")
    print(f"Rows: {len(result['clean']):,}")
    print(f"Encoded features: {result['encoded'].shape[1] - 1:,}")

    report_path = generate_report(result)

    print("\nBaseline analytics report complete.")
    print(f"Report: {report_path}")
    print("\nGenerated files are in data/processed/ and reports/")


if __name__ == "__main__":
    main()
