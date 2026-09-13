# Day 6-7: Missing Values, Encoding & Baseline Analytics
#
# Run the full pipeline from the repository root:
#     python -m src.run_pipeline
#
# This file is intentionally a Python script so it can be reviewed directly on GitHub.
# The reusable implementation is in src/.

from src.preprocessing import run_preprocessing
from src.baseline_report import generate_report

result = run_preprocessing()

print("Missing values before/after:")
print(result["missing"])

print("\nNumeric columns:")
print(result["numeric_cols"])

print("\nCategorical columns:")
print(result["categorical_cols"])

print("\nCleaned dataset preview:")
print(result["clean"].head())

print("\nEncoded dataset shape:")
print(result["encoded"].shape)

report = generate_report(result)
print(f"\nBaseline report created: {report}")
