# Week 2: Feature Engineering & Predictive Modeling

This directory contains the complete implementation for **Week 2** of the Customer Churn Prediction & Lifetime Value (LTV) Engine.

---

## 🎯 Week 2 Objectives & Completed Tasks

- [x] **Feature Engineering**: Engineered 47 domain-specific features capturing customer lifecycle cohorts, spending dynamics, bill shock, and service bundles.
- [x] **Classification Modeling**: Built, trained, and tuned **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers with class-imbalance weighting.
- [x] **Evaluation Benchmarking**: Evaluated models using **Precision**, **Recall**, **F1-Score**, **Accuracy**, and **ROC-AUC**.
- [x] **SHAP Explainability**: Implemented SHAP `TreeExplainer` for global feature importance (beeswarm & bar plots) and local individual customer waterfall explanations.
- [x] **Model & Report Artifacts**: Serialized champion models, preprocessing scalers, and generated publication-quality evaluation plots.

---

## 📂 Directory Structure

```text
week_2/
├── __init__.py
├── README.md                           # Comprehensive documentation & findings
├── data_loader.py                      # Dataset caching & loading pipeline
├── feature_engineering.py              # Feature creation & preprocessing logic
├── train_and_evaluate.py               # Model training, evaluation & metrics
├── shap_explainability.py              # SHAP beeswarm & waterfall explanations
├── run_week_2.py                       # Single-command orchestrator for Week 2
├── Week_2_Predictive_Modeling.ipynb    # Interactive Jupyter Notebook
├── data/
│   └── Telco-Customer-Churn.csv        # Real IBM Telco churn dataset (7,043 rows)
├── models/
│   ├── best_churn_model.pkl            # Champion model binary
│   ├── scaler.pkl                      # Fitted StandardScaler
│   └── model_metadata.json             # Feature names and benchmark scores
└── reports/
    ├── model_benchmark_results.csv     # Tabular metric comparison
    ├── model_metrics_comparison.png    # Bar chart of model performance
    ├── confusion_matrices_comparison.png # Side-by-side confusion matrix heatmaps
    ├── roc_curves_comparison.png       # Combined ROC curves with AUC
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

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** (Champion) | **77.15%** | **0.5483** | **78.88%** | **0.6469** | **0.8471** |
| **XGBoost** | 75.80% | 0.5303 | 77.27% | 0.6289 | 0.8455 |
| **Logistic Regression** | 73.46% | 0.5000 | 79.41% | 0.6136 | 0.8483 |

> **Key Takeaway**: Random Forest and XGBoost achieve an outstanding **~79% Recall** and **~0.85 ROC-AUC**. For proactive customer retention, prioritizing **Recall and F1-Score** ensures the business catches approximately 8 out of every 10 churning customers before cancellation occurs.

---

## 🔍 SHAP Explainability & Key Drivers

SHAP analysis using `TreeExplainer` reveals the top factors driving churn:

1. **`is_month_to_month`**: Customers on month-to-month contracts have the highest churn hazard. Moving them to 1- or 2-year contracts significantly mitigates churn.
2. **`tenure`**: Churn hazard decreases sharply as tenure exceeds 12–24 months.
3. **`bill_shock`**: Sudden billing increases relative to historical spend drive immediate cancellation behavior.
4. **`InternetService_Fiber optic`**: Fiber optic users churn at higher rates when not paired with tech support.
5. **`is_electronic_check`**: Payment method with the highest churn correlation compared to automated credit card or bank transfer.

---

## 🚀 How to Run

To execute the entire Week 2 pipeline end-to-end and regenerate all models and plots:

```bash
python week_2/run_week_2.py
```

To run individual components:
```bash
# Test feature engineering
python week_2/feature_engineering.py

# Train and evaluate models
python week_2/train_and_evaluate.py

# Generate SHAP plots
python week_2/shap_explainability.py
```
