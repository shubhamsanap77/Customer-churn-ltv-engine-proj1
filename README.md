# Customer Churn Prediction & Lifetime Value (LTV) Engine

A production-oriented data analytics project for customer churn prediction
and lifetime value analysis.

## Project Overview

The project uses customer demographic, service, tenure, and billing
information to:

- Identify customers at risk of churn
- Evaluate classification models for churn prediction
- Estimate customer lifetime value (LTV)
- Integrate churn predictions with customer-level LTV information
- Produce validated outputs for downstream analytics and dashboard/API work

## Data Source

### Telco Customer Churn Dataset

The project is based on the Telco Customer Churn dataset and uses customer
attributes including:

- Tenure
- Monthly charges
- Contract type
- Internet service
- Customer service features
- Churn status

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- SQL / PostgreSQL
- SQLAlchemy
- FastAPI
- Apache Superset / Metabase
- Git / GitHub

## Project Work Completed

### Data Preparation and Feature Engineering

The repository contains data preparation, preprocessing, feature engineering,
and baseline analytics outputs.

### Churn Model Development

The project contains saved churn classification pipelines for:

- Logistic Regression
- Decision Tree
- Random Forest

The saved scikit-learn pipelines include their required preprocessing steps.

### Churn + LTV Integration

The Week 4 Day 3 integration combines customer-level LTV information with
churn predictions and probabilities from the saved churn models.

The integrated output contains 5,174 customer records and preserves unique
customer IDs.

### Final Model Evaluation

The Week 4 Day 7 final evaluation independently verifies Accuracy, Precision,
Recall, and F1 Score using the verified Week 4 Day 2 prediction output.

Final reported model results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8034 | 0.6520 | 0.5562 | 0.6003 | 0.8417 |
| Decision Tree | 0.7928 | 0.6385 | 0.5053 | 0.5642 | 0.8376 |
| Random Forest | 0.7431 | 0.9286 | 0.0348 | 0.0670 | 0.8201 |

Based on the measured evaluation metrics, Logistic Regression is the
strongest overall performer among these three models.

Random Forest has the highest precision but substantially lower recall.

## Week 4 Day 7 Validation

The final Week 4 validation confirmed:

- 5,174 integrated customer records
- 5,174 unique customer IDs
- 0 missing customer IDs
- 0 missing Estimated_LTV values
- Churn predictions restricted to 0/1
- Churn probabilities within the [0, 1] range
- 0 missing values in the integrated output
- All dedicated Week 4 Day 7 tests passed
- All required Python dependencies were available
- `pip check` reported no broken requirements

## Important Output Locations

### Week 4 Day 3 Integration

```text
data/processed/week4_day3_integration/