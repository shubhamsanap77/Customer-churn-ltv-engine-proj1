# Week 2: Feature Engineering & Predictive Modeling

This directory contains the production-grade implementation for **Week 2** of the Customer Churn Prediction & Lifetime Value (LTV) Engine.

---

## 🎯 Week 2 Objectives & Completed Deliverables

- [x] **Feature Engineering**: Engineered 47 domain-specific features capturing customer lifecycle cohorts, spending dynamics, bill shock, and service bundles.
- [x] **Classification Modeling**: Built, trained, and tuned **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers with class-imbalance weighting.
- [x] **Evaluation Benchmarking**: Evaluated models using **Precision**, **Recall**, **F1-Score**, **Accuracy**, and **ROC-AUC**.
- [x] **SHAP Explainability**: Implemented SHAP `TreeExplainer` for global feature importance (beeswarm & bar plots) and local individual customer waterfall explanations.
- [x] **Hyperparameter Optimization**: 5-Fold Stratified Cross-Validation grid search on tree depths, estimators, and learning rates.
- [x] **Decision Threshold & Profit Optimization**: Analyzed the full threshold curve [0.05 - 0.95] and optimized retention campaign ROI.
- [x] **Automated Pipeline Tests**: 7-stage unit test suite verifying zero target leakage, clean imputation, and valid probability bounds.
- [x] **Model Governance**: Formal Google-standard [`MODEL_CARD.md`](./MODEL_CARD.md) including subgroup fairness analysis (Gender & Senior Citizen cohorts).

---

## 📂 Directory Structure

```text
week_2/
├── __init__.py
├── README.md                           # Week 2 documentation & findings
├── MODEL_CARD.md                       # Industry standard ML Model Card & fairness checks
├── data_loader.py                      # Dataset caching & loading pipeline
├── feature_engineering.py              # Feature creation & preprocessing logic
├── train_and_evaluate.py               # Model training, evaluation & metrics
├── hyperparameter_tuning.py            # 5-fold Stratified CV hyperparameter search
├── threshold_optimizer.py              # Classification threshold & business profit optimizer
├── shap_explainability.py              # SHAP beeswarm & waterfall explanations
├── test_pipeline.py                    # Automated unit & data integrity test suite
├── run_week_2.py                       # Master orchestrator executing entire pipeline
├── Week_2_Predictive_Modeling.ipynb    # Interactive Jupyter Notebook
├── data/
│   └── Telco-Customer-Churn.csv        # Real IBM Telco churn dataset (7,043 rows)
├── models/
│   ├── best_churn_model.pkl            # Champion model binary
│   ├── tuned_churn_model.pkl           # Cross-validated tuned model
│   ├── scaler.pkl                      # Fitted StandardScaler
│   └── model_metadata.json             # Feature names and benchmark scores
└── reports/
    ├── model_benchmark_results.csv     # Tabular metric comparison
    ├── tuning_results.csv              # CV hyperparameter search results
    ├── optimal_threshold_config.json   # Profit & F1 threshold configurations
    ├── model_metrics_comparison.png    # Bar chart of model performance
    ├── confusion_matrices_comparison.png # Side-by-side confusion matrix heatmaps
    ├── roc_curves_comparison.png       # Combined ROC curves with AUC
    ├── cv_hyperparameter_performance.png # CV vs Test F1 performance
    ├── precision_recall_threshold_curve.png # Precision-Recall vs threshold
    ├── profit_curve_by_threshold.png   # Net retention business value curve
    ├── shap_summary_beeswarm.png       # Global feature impact beeswarm
    ├── shap_feature_importance_bar.png # Mean |SHAP| ranking
    ├── shap_waterfall_high_risk.png    # Local high-risk customer explanation
    ├── shap_waterfall_low_risk.png     # Local low-risk customer explanation
    └── top_shap_churn_drivers.json     # Ranked churn factors in JSON
```

---

## 🧪 Engineered Features

| Feature Name | Category | Business Logic & Justification |
|---|---|---|
| `tenure_cohort` | Lifecycle | Binned into `0-12m`, `12-24m`, `24-48m`, `48-60m`, `60m+` to capture onboarding vs loyalty churn. |
| `monthly_to_total_ratio` | Financial | Ratio of monthly charges to cumulative spend; high values indicate high financial velocity for new users. |
| `avg_historical_monthly` | Financial | Historical average spend calculated as `TotalCharges / (tenure + 1)`. |
| `bill_shock` | Financial | Difference `MonthlyCharges - avg_historical_monthly`. Detects sudden price hikes that trigger churn. |
| `service_bundle_count` | Engagement | Count of active add-on services (`OnlineSecurity`, `OnlineBackup`, `TechSupport`, etc.). |
| `has_security_bundle` | Protection | Binary flag for customers with both `OnlineSecurity` and `TechSupport` (high retention cohort). |
| `fiber_without_support` | Friction | Fiber optic users without tech support experiencing network friction. |
| `is_month_to_month` | Contract | Strongest churn risk indicator in telecommunications. |
| `is_electronic_check` | Payment | Captures payment friction associated with manual electronic checks. |
| `has_family` | Demographic | Indicator if customer has a partner or dependents, representing household stickiness. |

---

## 📊 Model Evaluation Benchmark

Evaluation on stratified test set (1,409 customers, 26.54% churn rate):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
|---|---|---|---|---|---|---|
| **Random Forest** | **77.15%** | **0.5483** | **78.88%** | **0.6469** | **0.8471** | 🏆 Champion |
| **XGBoost** | 75.80% | 0.5303 | 77.27% | 0.6289 | 0.8455 | High Performer |
| **Logistic Regression** | 73.46% | 0.5000 | 79.41% | 0.6136 | 0.8483 | Baseline |

---

## 💰 Business Profit & Threshold Optimization

Default classification cutoffs (0.50) assume equal cost of false positives and false negatives. In customer churn, failing to identify a churner incurs heavy customer replacement acquisition costs ($500+), while retention outreach costs only $50.

| Operating Threshold | Precision | Recall | F1-Score | Expected Net Value |
|---|---|---|---|---|
| **0.50** (Default) | 54.83% | 78.88% | 0.6469 | $10,090.00 |
| **0.13** (Optimal Profit) | 35.80% | **98.40%** | 0.5242 | **$44,620.00** |

> Operating at the profit-optimal threshold of **0.13** captures **98.4% of churning accounts**, generating a projected **+$34,530 net value gain** over the default threshold.

---

## 🔍 SHAP Explainability & Key Drivers

1. **`is_month_to_month`**: Customers on month-to-month contracts have the highest churn hazard. Moving them to 1- or 2-year contracts significantly mitigates churn.
2. **`tenure`**: Churn hazard decreases sharply as tenure exceeds 12–24 months.
3. **`bill_shock`**: Sudden billing increases relative to historical spend drive immediate cancellation behavior.
4. **`InternetService_Fiber optic`**: Fiber optic users churn at higher rates when not paired with tech support.
5. **`is_electronic_check`**: Payment method with the highest churn correlation compared to automated credit card or bank transfer.

---

## 🚀 How to Run

To execute the entire Week 2 pipeline end-to-end:

```bash
python week_2/run_week_2.py
```

To run individual modules:
```bash
# Run automated pipeline tests
python week_2/test_pipeline.py

# Run hyperparameter tuning (5-fold CV)
python week_2/hyperparameter_tuning.py

# Run threshold & profit optimization
python week_2/threshold_optimizer.py
```
