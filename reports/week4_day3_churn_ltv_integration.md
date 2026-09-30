# Week 4 Day 3 – Churn + LTV Model Integration

## Objective

Integrate the Week 4 Day 2 churn classification models with the Week 4 Day 1 customer-level LTV dataset.

The integration generates churn predictions and churn probabilities for the same customer records that contain `Estimated_LTV`.

## Input

### LTV Dataset

File:

`data/processed/week4_day1_ltv_customer_value.csv`

Encoding:

`UTF-16`

Records integrated:

`5,174`

Customer identifier:

`customerID`

LTV field:

`Estimated_LTV`

### Churn Models

The following saved Week 4 Day 2 pipelines were used:

- Logistic Regression
- Decision Tree
- Random Forest

Each saved model contains its own preprocessing pipeline and was loaded using
scikit-learn version 1.6.1, matching the model training environment.

## Integration Method

The LTV dataset was used as the customer-level base.

The three saved churn pipelines were applied directly to the 20 input features expected by the models:

- customerID
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

No manual preprocessing was introduced because the saved models already contain
their required preprocessing steps.

The integration preserves `customerID` and combines:

- Actual churn value for reference
- Estimated LTV
- Logistic Regression churn prediction
- Logistic Regression churn probability
- Decision Tree churn prediction
- Decision Tree churn probability
- Random Forest churn prediction
- Random Forest churn probability
- Churn model agreement count
- Indicator showing whether all three models predict churn

## Validation

The generated integration output was validated successfully.

| Validation | Result |
|---|---:|
| Integrated customer rows | 5,174 |
| Unique customer IDs | 5,174 |
| Duplicate customer IDs | 0 |
| Missing customer IDs | 0 |
| Missing Estimated_LTV values | 0 |
| Output columns | 11 |

### Estimated LTV

| Metric | Value |
|---|---:|
| Minimum | 19.10 |
| Maximum | 7,448.55 |
| Mean | 1,866.85 |

### Churn predictions

| Model | Predicted Churn |
|---|---:|
| Logistic Regression | 299 |
| Decision Tree | 486 |
| Random Forest | 8 |

All three models produced predictions for all 5,174 integrated customers.

## Output Files

Main customer-level integration:

`data/processed/week4_day3_integration/churn_ltv_integrated_customer_output.csv`

Integration validation summary:

`data/processed/week4_day3_integration/integration_summary.csv`

## Result

The Week 4 Day 3 integration successfully connects the customer-level
`Estimated_LTV` information with churn predictions and probabilities from all
three Week 4 Day 2 classification pipelines.

The resulting dataset is suitable as an integrated output for subsequent
dashboard/API work, subject to the team's further model and business validation.