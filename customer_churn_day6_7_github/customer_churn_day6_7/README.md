# Customer Churn Prediction – Day 6-7

This repository implements **Day 6-7 of the Customer Churn Prediction & LTV project**:

- Handle missing values
- Convert numeric fields safely
- Encode categorical variables
- Produce a cleaned dataset
- Establish a baseline churn analytics report
- Generate summary tables and baseline visualizations

The project is designed for the standard **Telco Customer Churn** dataset.

## Project structure

```text
customer_churn_day6_7/
├── data/
│   ├── raw/                  # Put the Telco CSV here
│   └── processed/            # Generated cleaned/encoded data
├── reports/                  # Generated baseline report and charts
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── baseline_report.py
│   └── run_pipeline.py
├── notebooks/
│   └── day6_7_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Add the dataset

Download the Telco Customer Churn CSV and place it in:

```text
data/raw/
```

The script automatically searches for the first `.csv` file in that folder.

Typical filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

**Do not upload the dataset to GitHub** if your project rules do not permit redistribution. The `.gitignore` already excludes CSV files inside `data/raw`.

## 3. Run Day 6-7 pipeline

From the repository root:

```bash
python -m src.run_pipeline
```

The pipeline will:

1. Load the raw CSV.
2. Standardize column names.
3. Convert `TotalCharges` to numeric.
4. Detect missing values.
5. Impute numeric missing values using the median.
6. Impute categorical missing values using the most frequent value.
7. Encode categorical variables using one-hot encoding.
8. Convert `Churn` to a binary target.
9. Save the cleaned encoded dataset.
10. Generate baseline churn analytics.
11. Generate charts and a Markdown report.

## 4. Output files

After running the pipeline:

```text
data/processed/telco_churn_cleaned.csv
data/processed/telco_churn_encoded.csv
reports/baseline_analytics_report.md
reports/missing_values_before_after.csv
reports/categorical_summary.csv
reports/numeric_summary.csv
reports/churn_by_category.csv
reports/churn_distribution.png
reports/churn_by_contract.png
reports/churn_by_tenure_group.png
```

## Baseline analytics included

The baseline report contains:

- Dataset dimensions
- Duplicate count
- Missing-value analysis before/after preprocessing
- Churn count and churn rate
- Numeric feature statistics
- Categorical feature distributions
- Churn rate by contract type
- Churn rate by internet service
- Churn rate by payment method
- Churn rate by tenure group
- Churn rate by senior-citizen status
- Key baseline observations

## Important

This is **preprocessing + baseline analytics**, not the Week 2 predictive-modeling stage. Logistic Regression, Random Forest, XGBoost and SHAP should be added in the later stages of the project.
