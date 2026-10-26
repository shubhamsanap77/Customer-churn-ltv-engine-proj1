# Day 6 — XGBoost Training & Model Comparison

## Scope

Week 2 Day 4–6 requires Logistic Regression, Random Forest, and XGBoost, evaluated with precision, recall, and F1-score.

## Data used

- Training features: `5,634 rows × 5,335 features`
- Test features: `1,409 rows × 5,335 features`
- Training churn rate: `0.2654`
- Test churn rate: `0.2654`
- Random state: `42`
- XGBoost `scale_pos_weight`: `2.7686`

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7530 | 0.5241 | 0.7567 | 0.6193 | 0.8395 | 0.47 |
| XGBoost | 0.7502 | 0.5200 | 0.7647 | 0.6190 | 0.8387 | 5.55 |
| Random Forest | 0.7012 | 0.4638 | 0.8048 | 0.5885 | 0.8065 | 0.62 |

## Artifacts

- `data/processed/day6_model_comparison/model_metrics.csv`
- `data/processed/day6_model_comparison/classification_reports.txt`
- `data/processed/day6_model_comparison/xgboost_feature_importance.csv`
- `data/processed/day6_model_comparison/model_comparison.png`
- `data/processed/day6_model_comparison/model_config.json`

## Notes

All three models use the same Day 3 train/test split. The requested precision, recall, and F1 metrics are reported on the held-out test set; accuracy and ROC-AUC are included as additional diagnostics.
