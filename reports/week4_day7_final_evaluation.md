# Week 4 Day 7 – Final Model Evaluation, Documentation & GitHub Cleanup

## Objective

Complete the final evaluation and validation of the Week 4 customer churn
models, document the verified results, validate the Week 4 Churn + LTV
integration, and prepare the repository for final team use.

## Final Evaluation Scope

The final evaluation uses the verified Week 4 Day 2 held-out prediction
results and validates the Week 4 Day 3 customer-level Churn + LTV integration.

The Week 4 Day 2 evaluation contains:

- 1,409 held-out test records
- 1,035 non-churn records
- 374 churn records

The Week 4 Day 3 integrated customer-level output contains:

- 5,174 customers
- 5,174 unique customer IDs

## Model Evaluation

The following Week 4 Day 2 churn models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

### Final Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8034 | 0.6520 | 0.5562 | 0.6003 | 0.8417 |
| Decision Tree | 0.7928 | 0.6385 | 0.5053 | 0.5642 | 0.8376 |
| Random Forest | 0.7431 | 0.9286 | 0.0348 | 0.0670 | 0.8201 |

### Model Comparison

Logistic Regression achieved the highest:

- Accuracy: 0.8034
- F1 Score: 0.6003
- ROC-AUC: 0.8417

Random Forest achieved the highest precision at 0.9286, but its recall was
only 0.0348, meaning it detected very few churn cases.

Based on the measured evaluation metrics, Logistic Regression is the
strongest overall model among the three evaluated models.

Final model selection should still consider the business cost of false
positives versus false negatives before production deployment.

## Independent Verification

Accuracy, Precision, Recall, and F1 Score were independently recalculated
from the Week 4 Day 2 prediction file.

The independently calculated values matched the reported Week 4 Day 2
reference values for all three models.

ROC-AUC was not independently recalculated because the available Week 4 Day 2
prediction file contains hard class predictions rather than probability or
decision scores. The ROC-AUC values are therefore retained from the original
Week 4 Day 2 model comparison output.

## Week 4 Day 3 Churn + LTV Integration Validation

The final Week 4 Day 3 customer-level integration output was validated before
completion of the Day 7 work.

Validation results:

| Check | Result |
|---|---:|
| Integrated customer rows | 5,174 |
| Unique customer IDs | 5,174 |
| Missing customer IDs | 0 |
| Missing Estimated_LTV values | 0 |
| Estimated_LTV data type | float64 |
| Minimum Estimated_LTV | 19.10 |
| Prediction values outside 0/1 | 0 |
| Probability values outside [0,1] | 0 |
| Missing values in integrated output | 0 |

All integration validation checks passed.

## Final Output Artifacts

The final Week 4 Day 7 evaluation generated:

- `data/processed/week4_day7_final_evaluation/final_model_evaluation.csv`
- `data/processed/week4_day7_final_evaluation/final_classification_reports.txt`
- `data/processed/week4_day7_final_evaluation/final_validation_summary.csv`
- `data/processed/week4_day7_final_evaluation/final_evaluation_summary.txt`
- `data/processed/week4_day7_final_evaluation/final_model_comparison.png`

Reference evidence used for the final verification:

- `data/processed/week4_day7_final_evaluation/reference/week4_day2_model_comparison.csv`
- `data/processed/week4_day7_final_evaluation/reference/week4_day2_predictions.csv`

## Evaluation Script

Final evaluation script:

`src/week4_day7_final_evaluation.py`

The script was syntax-checked successfully and executed successfully.

Execution result:

`SUCCESS: Week 4 Day 7 final evaluation completed.`

## Testing

A dedicated Week 4 Day 7 test suite was added:

`tests/test_week4_day7_final.py`

Test execution:

```text
python -m unittest tests.test_week4_day7_final -v