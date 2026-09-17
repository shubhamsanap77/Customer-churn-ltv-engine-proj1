# Week 2 — Day 4: Train Logistic Regression and Evaluate

**Assigned Contributor**: Rajarsh  
**Task**: Train Logistic Regression and evaluate using **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## 🎯 Task Objectives

- [x] Load and preprocess the **Telco Customer Churn** dataset (7,043 customer accounts).
- [x] Apply feature engineering: lifecycle cohorts, billing dynamics (`bill_shock`, `monthly_to_total_ratio`), service bundles, and numerical standard scaling.
- [x] Train **Standard Logistic Regression** and **Class-Balanced Logistic Regression** (`class_weight='balanced'`).
- [x] Evaluate and compare models on a stratified 20% test holdout ($N = 1,409$).
- [x] Generate confusion matrices, ROC curves, metric comparisons, and save model artifacts.

---

## 📊 Evaluation Results (Precision, Recall, F1-Score)

| Model Variant | Precision | Recall | F1-Score | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Standard)** | **0.6733** | 0.5401 | 0.5994 | **80.84%** | 0.8474 |
| **Logistic Regression (Balanced)** | 0.4983 | **0.7968** | **0.6132** | 73.31% | **0.8483** |

### 💡 Key Findings & Evaluation Analysis
1. **Recall Priority for Churn Prevention**:
   - In customer churn detection, **Recall** is the most critical operational metric because missing a churner costs significantly more than sending an unnecessary retention perk.
   - The **Class-Balanced Logistic Regression** model achieves **79.68% Recall**, capturing ~80% of all churners before they cancel.
2. **Balanced F1-Score**:
   - The balanced model delivers an improved **F1-Score of 0.6132** (compared to 0.5994 for standard).
3. **Discriminative Ability**:
   - Both variants achieve an **ROC-AUC of ~0.848**, indicating strong rank-ordering separation between churners and non-churners.

---

## 📂 Deliverables & File Structure

```text
week_2/
├── __init__.py
├── README.md                                      # Task documentation and evaluation findings
├── data_loader.py                                 # Dataset loading & caching logic
├── feature_engineering.py                         # Preprocessing and feature engineering
├── train_logistic_regression.py                   # Main Day 4 execution script
├── Day_4_Logistic_Regression_Evaluation.ipynb     # Interactive Jupyter Notebook
├── data/
│   └── Telco-Customer-Churn.csv                   # Raw dataset
├── models/
│   ├── logistic_regression_model.pkl              # Serialized balanced model binary
│   ├── scaler.pkl                                 # Fitted StandardScaler
│   └── model_metadata.json                        # Metadata and benchmark scores
└── reports/
    ├── logistic_regression_evaluation.csv         # Tabular benchmark metrics
    ├── logistic_regression_metrics.png            # Bar chart of Precision, Recall, F1, ROC-AUC
    ├── logistic_regression_confusion_matrix.png   # Confusion matrices
    └── logistic_regression_roc_curve.png          # ROC curve
```

---

## 🚀 How to Run

Execute the pipeline with:

```bash
python week_2/train_logistic_regression.py
```
