from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLEANED_FILE = PROCESSED_DIR / "telco_churn_cleaned.csv"
ENCODED_FILE = PROCESSED_DIR / "telco_churn_encoded.csv"
BASELINE_REPORT = REPORTS_DIR / "baseline_analytics_report.md"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
