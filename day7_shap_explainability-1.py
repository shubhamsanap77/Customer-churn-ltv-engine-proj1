# ============================================================
# WEEK 2 - DAY 7
# Customer Churn Prediction & LTV Engine
# SHAP Explainability + Business Interpretation
# ============================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -----------------------------
# 1. PATHS
# -----------------------------
# Project structure expected:
# Customer-churn-ltv-engine-proj1/
# ├── data/
# │   ├── raw/
# │   └── processed/
# └── notebooks/

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROCESSED_DIR / "day7_shap"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Search common dataset locations automatically
possible_files = [
    DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    DATA_DIR / "processed" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

DATA_PATH = next((p for p in possible_files if p.exists()), None)

if DATA_PATH is None:
    csv_files = list(DATA_DIR.rglob("*.csv"))
    if csv_files:
        # Prefer files containing "churn"
        churn_files = [p for p in csv_files if "churn" in p.name.lower()]
        DATA_PATH = churn_files[0] if churn_files else csv_files[0]

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Put your churn CSV inside the project's data/ "
        "or data/raw/ folder."
    )

print(f"Dataset loaded from: {DATA_PATH}")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# 3. CLEAN COLUMN NAMES
# -----------------------------
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# -----------------------------
# 4. FIND TARGET COLUMN
# -----------------------------
target_candidates = [
    "Churn",
    "churn",
    "CHURN",
    "Churn_Label",
    "churn_label",
    "Churn_Value",
    "Churn_Value",
    "churn_value",
]

target = next((c for c in target_candidates if c in df.columns), None)

if target is None:
    raise KeyError(
        "Churn target column not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )

print(f"\nTarget column: {target}")

# -----------------------------
# 5. CONVERT TARGET TO 0/1
# -----------------------------
def convert_target(series):
    s = series.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "churn": 1,
        "not churn": 0,
    }

    mapped = s.map(mapping)

    # If mapping did not work, try numeric conversion
    if mapped.notna().sum() == 0:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() > 0:
            mapped = numeric

    return mapped

y = convert_target(df[target])

valid = y.notna()
df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

# -----------------------------
# 6. REMOVE TARGET + ID COLUMNS
# -----------------------------
X = df.drop(columns=[target], errors="ignore").copy()

# Remove common identifier columns because they should not drive churn
id_columns = [
    "customerid",
    "customer_id",
    "CustomerID",
    "Customer_Id",
    "id",
    "ID",
]

drop_ids = [c for c in id_columns if c in X.columns]
if drop_ids:
    X = X.drop(columns=drop_ids)
    print("\nRemoved identifier columns:", drop_ids)

# -----------------------------
# 7. HANDLE NUMERIC-LIKE COLUMNS
# -----------------------------
# Convert TotalCharges when it contains spaces/strings
for col in X.columns:
    if X[col].dtype == "object":
        cleaned = X[col].astype(str).str.strip()
        numeric_version = pd.to_numeric(cleaned, errors="coerce")

        # Convert only if most non-null values are numeric
        non_empty = cleaned.ne("").sum()
        if non_empty > 0:
            numeric_ratio = numeric_version.notna().sum() / non_empty
            if numeric_ratio >= 0.90:
                X[col] = numeric_version

# -----------------------------
# 8. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in X_train.columns
    if c not in numeric_features
]

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# -----------------------------
# 9. PREPROCESSING
# -----------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# -----------------------------
# 10. RANDOM FOREST MODEL
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model),
])

print("\nTraining model...")
pipeline.fit(X_train, y_train)

# -----------------------------
# 11. MODEL EVALUATION
# -----------------------------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")

with open(OUTPUT_DIR / "model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("WEEK 2 - DAY 7 MODEL METRICS\n")
    f.write("=" * 40 + "\n\n")
    f.write(classification_report(y_test, y_pred))
    f.write(f"\nROC-AUC: {auc:.4f}\n")

# -----------------------------
# 12. TRANSFORM FEATURES
# -----------------------------
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# Clean feature names for readability
feature_names = np.array([
    name.replace("num__", "")
        .replace("cat__", "")
        .replace("onehot__", "")
    for name in feature_names
])

print("\nTransformed feature count:", len(feature_names))

# Train RF directly on transformed data for SHAP compatibility
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

rf_model.fit(X_train_transformed, y_train)

# -----------------------------
# 13. SHAP
# -----------------------------
try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP is not installed. Run:\n"
        "pip install shap"
    )

print("\nCalculating SHAP values...")

# TreeExplainer works with Random Forest
explainer = shap.TreeExplainer(rf_model)

# Use a sample for faster plots on large datasets
sample_size = min(1000, X_test_transformed.shape[0])

rng = np.random.RandomState(42)
sample_indices = rng.choice(
    X_test_transformed.shape[0],
    size=sample_size,
    replace=False
)

X_shap = X_test_transformed[sample_indices]
y_shap = y_test.iloc[sample_indices].reset_index(drop=True)

shap_values = explainer.shap_values(X_shap)

# SHAP versions may return list or ndarray
if isinstance(shap_values, list):
    if len(shap_values) == 2:
        shap_positive = shap_values[1]
    else:
        shap_positive = shap_values[0]
else:
    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        # Usually: samples x features x classes
        shap_positive = shap_array[:, :, 1]
    else:
        shap_positive = shap_array

# -----------------------------
# 14. SHAP SUMMARY PLOT
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_positive,
    X_shap,
    feature_names=feature_names,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# -----------------------------
# 15. SHAP BAR PLOT
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_positive,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

# -----------------------------
# 16. FEATURE IMPORTANCE TABLE
# -----------------------------
mean_abs_shap = np.abs(shap_positive).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
}).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTop 20 SHAP features:")
print(importance_df.head(20).to_string(index=False))

# -----------------------------
# 17. INDIVIDUAL CUSTOMER EXPLANATION
# -----------------------------
customer_position = 0

customer_original = X_test.iloc[sample_indices[customer_position]].copy()
customer_probability = y_prob[sample_indices[customer_position]]
customer_actual = y_test.iloc[sample_indices[customer_position]]

customer_shap = shap_positive[customer_position]

customer_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": customer_shap,
    "Absolute_SHAP": np.abs(customer_shap)
}).sort_values(
    "Absolute_SHAP",
    ascending=False
)

customer_explanation.to_csv(
    OUTPUT_DIR / "individual_customer_shap.csv",
    index=False
)

# -----------------------------
# 18. BUSINESS INSIGHTS
# -----------------------------
top_features = importance_df.head(10)

insights = []

insights.append("WEEK 2 - DAY 7 BUSINESS INTERPRETATION")
insights.append("=" * 50)
insights.append("")
insights.append(
    f"Model ROC-AUC on the test set: {auc:.4f}"
)
insights.append(
    "SHAP identifies which features contribute most to the model's churn predictions."
)
insights.append("")
insights.append("Top churn-prediction drivers:")

for i, row in top_features.iterrows():
    insights.append(
        f"- {row['Feature']}: mean absolute SHAP = "
        f"{row['Mean_Absolute_SHAP']:.6f}"
    )

insights.append("")
insights.append("Individual customer explanation:")
insights.append(
    f"- Predicted churn probability: {customer_probability:.2%}"
)
insights.append(
    f"- Actual churn label: {int(customer_actual)}"
)
insights.append("")
insights.append(
    "Interpretation rule: positive SHAP values push the model toward churn "
    "for the positive/churn class; negative values push the prediction away "
    "from churn."
)
insights.append("")
insights.append(
    "Business use: the identified drivers can be used to prioritize "
    "customer-retention analysis and targeted interventions."
)

with open(
    OUTPUT_DIR / "business_insights.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(insights))

# -----------------------------
# 19. SAVE SAMPLE PREDICTIONS
# -----------------------------
predictions = X_test.copy()
predictions["Actual_Churn"] = y_test.values
predictions["Predicted_Churn"] = y_pred
predictions["Churn_Probability"] = y_prob

predictions.to_csv(
    OUTPUT_DIR / "test_customer_predictions.csv",
    index=False
)

# -----------------------------
# 20. FINAL STATUS
# -----------------------------
print("\n" + "=" * 60)
print("DAY 7 COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Output folder: {OUTPUT_DIR}")
print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nNext: add the generated Day 7 output files to GitHub.")