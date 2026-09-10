# Model Card: Customer Churn Prediction Engine

This Model Card follows the standard framework proposed by Mitchell et al. (2019) to ensure transparency, accountability, and reproducibility in production machine learning systems.

---

## 1. Model Details

- **Model Name**: Telco Customer Churn Classifier
- **Model Version**: 2.1.0
- **Model Type**: Supervised Ensemble Classification (Random Forest & XGBoost with class-imbalance weighting)
- **Frameworks**: `scikit-learn 1.8.0`, `xgboost 3.4.1`, `shap 0.52.0`, `pandas 3.0.0`
- **Release Date**: September 2026
- **License**: MIT
- **Primary Point of Contact**: Rajarsh & Data Science Team

---

## 2. Intended Use

- **Primary Intended Use**: Identification of subscription and telecommunications customers with a high probability of cancelling their service (churning) within the subsequent billing cycle.
- **Primary Users**: Customer Retention Specialists, Marketing Automation Engines, and Account Managers.
- **Out-of-Scope Use Cases**: 
  - Credit scoring, lending underwriting, or debt collection prioritization.
  - Punitive rate adjustments or targeted price increases.

---

## 3. Training & Evaluation Data

- **Dataset Source**: IBM Telco Customer Churn Benchmark Dataset
- **Total Population**: 7,043 customers across 21 raw attributes
- **Split Strategy**: Stratified 80/20 train/test split preserving the natural 26.54% churn rate
  - **Training Set**: 5,634 samples
  - **Test Set**: 1,409 samples
- **Features Engineered**: 47 numerical, categorical, and behavioral indicators:
  - *Lifecycle*: Tenure cohorts (`0-12m`, `12-24m`, `24-48m`, `48-60m`, `60m+`)
  - *Financial Dynamics*: `monthly_to_total_ratio`, `bill_shock`, `avg_historical_monthly`
  - *Bundling*: `service_bundle_count`, `has_security_bundle`, `has_streaming_bundle`
  - *Friction*: `fiber_without_support`, `is_month_to_month`, `is_electronic_check`

---

## 4. Benchmark Performance Metrics

Evaluated on held-out test data ($N=1,409$):

| Evaluation Metric | Baseline (Logistic Regression) | Random Forest (Champion) | XGBoost Classifier |
|---|---|---|---|
| **Accuracy** | 73.46% | **77.15%** | 75.80% |
| **Precision** | 50.00% | **54.83%** | 53.03% |
| **Recall (Sensitivity)** | **79.41%** | 78.88% | 77.27% |
| **F1-Score** | 61.36% | **64.69%** | 62.89% |
| **ROC-AUC Score** | 0.8483 | 0.8471 | 0.8455 |

*Note: Models were configured with positive class weighting ($\approx 2.77$) to deliberately optimize for high Recall ($\sim 79\%$), capturing the majority of churning accounts before contract termination.*

---

## 5. Subgroup Fairness & Demographic Parity Analysis

To verify equitable model performance across protected demographic groups, evaluation metrics were segmented across customer cohorts:

### A. Gender Fairness (Male vs. Female)
- **Female Test Cohort**: Recall = 78.6%, Precision = 54.2%, ROC-AUC = 0.846
- **Male Test Cohort**: Recall = 79.1%, Precision = 55.4%, ROC-AUC = 0.848
- **Disparate Impact Ratio**: $0.98$ (within the legal 0.80–1.25 four-fifths rule).

### B. Senior Citizen Cohort (Age Disparity Check)
- **Senior Citizens (Age 65+)**: Recall = 81.2%, Precision = 58.1%
- **Non-Senior Citizens**: Recall = 78.1%, Precision = 53.4%
- **Observation**: Senior citizens exhibit higher recall due to higher engagement with fixed line services; no negative bias detected.

---

## 6. Model Explainability (SHAP Insights)

SHAP (SHapley Additive exPlanations) values reveal the top factors influencing positive churn predictions:
1. **Contract Structure (`is_month_to_month`)**: Accounts with month-to-month contracts exhibit an average +0.63 log-odds increase in churn probability.
2. **Tenure Duration (`tenure`)**: Customers under 12 months are at highest risk; risk drops significantly after month 24.
3. **Bill Shock (`bill_shock`)**: Unexpected billing increases relative to historical averages are a major driver of dissatisfaction.
4. **Service Friction (`fiber_without_support`)**: High-bandwidth fiber customers without technical support bundle cancel at elevated rates.
5. **Payment Friction (`is_electronic_check`)**: Higher churn incidence compared to automatic bank or card payments.

---

## 7. Limitations & Recommendations

- **Cross-Sectional Limitation**: Data represents a single static cohort. Periodic retraining (monthly or quarterly) is required to capture macroeconomic shifts.
- **Decision Thresholding**: The default 0.50 cutoff is not financially optimal. Operating at a threshold of **0.38–0.42** yields maximum net business value for retention campaigns based on offer economics.
