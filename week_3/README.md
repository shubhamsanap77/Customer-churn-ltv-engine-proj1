# Week 3 — Day 4: Customer Lifetime Value (LTV) Prediction & Calculation Engine

**Assigned Contributor**: Rajarsh  
**Tasks**:
1. **LTV prediction/calculation module** (`week_3/ltv_module.py`)
2. **LTV calculation/prediction notebook + output** (`week_3/LTV_Calculation_and_Prediction.ipynb`)

---

## 🎯 Task Summary & Methodologies

This module implements comprehensive Customer Lifetime Value (LTV / CLV) analytics and machine learning regression for subscription telecommunications accounts.

### 1. Actuarial & Financial LTV Formulations
- **Historical LTV**: Cumulative realized billing revenue to date ($\text{TotalCharges} \approx \text{tenure} \times \text{MonthlyCharges}$).
- **Projected Future LTV**: Expected future revenue discounted by contract churn hazard rate and gross profit margin ($\text{gross margin} = 75\%$):
  $$\text{Future LTV} = \text{MonthlyCharges} \times \text{Expected Future Tenure (months)} \times \text{Gross Margin}$$
- **Total LTV**: $\text{Historical LTV} + \text{Projected Future LTV}$.

### 2. Strategic Customer Value Tiers (LTV Segmentation)
Customers are segmented into 4 actionable value cohorts based on lifetime value percentiles:

| LTV Tier | Definition | Customer Count | Mean Total LTV ($) | Revenue Contribution (%) |
|---|---|---|---|---|
| **Platinum (VIP)** | Top 20% | 1,409 | **$8,434.59** | **53.3%** |
| **Gold** | 60th - 80th percentile | 1,408 | **$4,146.98** | **26.2%** |
| **Silver** | 40th - 60th percentile | 1,409 | **$2,044.88** | **12.9%** |
| **Bronze** | Bottom 40% | 2,817 | **$606.50** | **7.7%** |

> **Key Business Finding**: **Platinum customers represent only 20% of the account base but drive 53.3% of total lifetime value.** Retaining Platinum accounts flagged as high churn risk yields the highest return on retention spend.

---

## 📊 Supervised Machine Learning LTV Regression Benchmark

Evaluated on a held-out test set ($N_{test} = 1,409$):

| Model | MAE ($) | RMSE ($) | $R^2$ Score | MAPE (%) | Recommendation |
|---|---|---|---|---|---|
| **XGBoost Regressor** | **$52.51** | **$78.45** | **0.9988** | 4.79% | 🏆 **Champion Model** |
| **Random Forest Regressor** | $53.51 | $80.96 | 0.9987 | **3.87%** | Ensemble Alternative |
| **Ridge Regression** | $566.57 | $699.74 | 0.9059 | 354.36% | Linear Baseline |

---

## 📂 Deliverables & File Structure

```text
week_3/
├── __init__.py
├── README.md                                      # Day 4 task documentation & benchmarks
├── ltv_module.py                                  # Core LTV calculation & regression module
├── run_ltv.py                                     # Standalone runner script
├── LTV_Calculation_and_Prediction.ipynb           # Executed Jupyter Notebook with outputs
├── data/
│   └── Telco-Customer-Churn.csv                   # Cleaned dataset
├── models/
│   ├── best_ltv_model.pkl                         # Serialized champion XGBoost model
│   ├── ltv_scaler.pkl                             # Fitted StandardScaler
│   └── ltv_metadata.json                          # Hyperparameters, metrics & features
└── reports/
    ├── ltv_evaluation_metrics.csv                 # Regression metrics benchmark table
    ├── ltv_actual_vs_predicted.png                # Actual vs Predicted scatter plot
    ├── ltv_distribution.png                       # LTV distribution by contract type
    ├── ltv_tier_segmentation.png                  # Customer count & revenue by LTV tier
    └── ltv_feature_importance.png                 # Top 10 feature importance rankings
```

---

## 🚀 How to Run

To run the full LTV pipeline:

```bash
python week_3/run_ltv.py
```
