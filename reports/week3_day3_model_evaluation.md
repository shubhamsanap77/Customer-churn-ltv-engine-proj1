# Week 3 Day 3 — Model Evaluation & Comparison

## Objective

Evaluate the churn classification models using Accuracy, Precision, Recall, F1-score, and ROC-AUC and provide a direct model comparison.

## Evaluation setup

- Training data: `5,634 rows × 5,335 features`
- Held-out test data: `1,409 rows × 5,335 features`
- Training churn rate: `0.2654`
- Test churn rate: `0.2654`
- Same held-out test set used for every model.
- Random state: `42`.
- XGBoost scale_pos_weight: `2.7686`.

## Model evaluation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7530 | 0.5241 | 0.7567 | 0.6193 | 0.8395 |
| Random Forest | 0.7012 | 0.4638 | 0.8048 | 0.5885 | 0.8065 |
| XGBoost | 0.7502 | 0.5200 | 0.7647 | 0.6190 | 0.8387 |

## Interpretation

The table reports the measured performance of all three models on the same held-out test set. No single metric is treated as a universal decision rule; the choice of model should depend on the business cost of false positives versus false negatives and the team's deployment requirements.

## Generated artifacts

- `data/processed/week3_day3_evaluation/model_evaluation_metrics.csv`
- `data/processed/week3_day3_evaluation/classification_reports.txt`
- `data/processed/week3_day3_evaluation/confusion_matrices.json`
- `data/processed/week3_day3_evaluation/model_comparison.png`
- `data/processed/week3_day3_evaluation/evaluation_summary.txt`
