from pathlib import Path

import joblib
import pandas as pd


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LTV_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "week4_day1_ltv_customer_value.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week4_day3_integration"

OUTPUT_CSV = OUTPUT_DIR / "churn_ltv_integrated_customer_output.csv"
SUMMARY_CSV = OUTPUT_DIR / "integration_summary.csv"


# -------------------------------------------------------------------
# Model files
# -------------------------------------------------------------------
MODEL_FILES = {
    "Logistic": MODEL_DIR / "week4_day2_logistic_regression.joblib",
    "DecisionTree": MODEL_DIR / "week4_day2_decision_tree.joblib",
    "RandomForest": MODEL_DIR / "week4_day2_random_forest.joblib",
}


# These are the exact 20 features expected by Shubham's saved models.
MODEL_FEATURES = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def load_ltv_dataset() -> pd.DataFrame:
    """Load and validate the Week 4 Day 1 LTV dataset."""

    if not LTV_INPUT.exists():
        raise FileNotFoundError(
            f"LTV input file not found: {LTV_INPUT}"
        )

    # The source file is UTF-16 encoded.
    df = pd.read_csv(LTV_INPUT, encoding="utf-16")

    required_columns = MODEL_FEATURES + ["Estimated_LTV"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Customer-level integrity checks.
    if df["customerID"].isna().any():
        raise ValueError("customerID contains missing values.")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values detected.")

    if df["Estimated_LTV"].isna().any():
        raise ValueError("Estimated_LTV contains missing values.")

    return df


def get_positive_probability(model, X: pd.DataFrame) -> pd.Series:
    """
    Return probability for the positive churn class.

    The saved models are expected to use a binary target containing 0/1.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Positive churn class 1 was not found. Model classes: {classes}"
        )

    positive_index = classes.index(1)

    probabilities = model.predict_proba(X)[:, positive_index]

    return pd.Series(probabilities, index=X.index)


def score_model(
    model_name: str,
    model_path: Path,
    X: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Load one model and generate churn prediction + probability."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    model = joblib.load(model_path)

    predictions = pd.Series(
        model.predict(X),
        index=X.index,
        dtype="int64",
    )

    probabilities = get_positive_probability(model, X)

    return predictions, probabilities


def build_integration() -> pd.DataFrame:
    """Generate the complete Churn + LTV customer-level output."""

    df = load_ltv_dataset()

    X = df[MODEL_FEATURES].copy()

    integrated = pd.DataFrame(
        {
            "customerID": df["customerID"],
            "Actual_Churn": df["Churn"] if "Churn" in df.columns else None,
            "Estimated_LTV": df["Estimated_LTV"],
        }
    )

    # Preserve the output from every Week 4 Day 2 churn model.
    for model_name, model_path in MODEL_FILES.items():

        predictions, probabilities = score_model(
            model_name,
            model_path,
            X,
        )

        integrated[f"{model_name}_Churn_Prediction"] = predictions
        integrated[f"{model_name}_Churn_Probability"] = probabilities

    # Useful integration-level indicator:
    # number of models predicting churn for each customer.
    prediction_columns = [
        f"{model_name}_Churn_Prediction"
        for model_name in MODEL_FILES
    ]

    integrated["Churn_Model_Agreement_Count"] = (
        integrated[prediction_columns].sum(axis=1)
    )

    # 3 = all models predict churn
    # 0 = no model predicts churn
    integrated["All_Models_Predict_Churn"] = (
        integrated["Churn_Model_Agreement_Count"] == len(MODEL_FILES)
    )

    # Final integrity checks.
    if len(integrated) != len(df):
        raise ValueError(
            "Integrated output row count does not match the LTV input."
        )

    if integrated["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customerID detected in integrated output."
        )

    probability_columns = [
        column
        for column in integrated.columns
        if column.endswith("_Churn_Probability")
    ]

    for column in probability_columns:
        if integrated[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if not integrated[column].between(0, 1).all():
            raise ValueError(
                f"{column} contains probabilities outside [0, 1]."
            )

    return integrated


def create_summary(integrated: pd.DataFrame) -> pd.DataFrame:
    """Create an auditable integration summary."""

    probability_columns = [
        "Logistic_Churn_Probability",
        "DecisionTree_Churn_Probability",
        "RandomForest_Churn_Probability",
    ]

    prediction_columns = [
        "Logistic_Churn_Prediction",
        "DecisionTree_Churn_Prediction",
        "RandomForest_Churn_Prediction",
    ]

    rows = []

    rows.append(
        {
            "Metric": "Integrated customer rows",
            "Value": len(integrated),
        }
    )

    rows.append(
        {
            "Metric": "Unique customer IDs",
            "Value": integrated["customerID"].nunique(),
        }
    )

    rows.append(
        {
            "Metric": "Missing Estimated_LTV",
            "Value": integrated["Estimated_LTV"].isna().sum(),
        }
    )

    rows.append(
        {
            "Metric": "Minimum Estimated_LTV",
            "Value": integrated["Estimated_LTV"].min(),
        }
    )

    rows.append(
        {
            "Metric": "Maximum Estimated_LTV",
            "Value": integrated["Estimated_LTV"].max(),
        }
    )

    for column in probability_columns:
        rows.append(
            {
                "Metric": f"{column} minimum",
                "Value": integrated[column].min(),
            }
        )

        rows.append(
            {
                "Metric": f"{column} maximum",
                "Value": integrated[column].max(),
            }
        )

    for column in prediction_columns:
        rows.append(
            {
                "Metric": f"{column} positive predictions",
                "Value": int(integrated[column].sum()),
            }
        )

    rows.append(
        {
            "Metric": "All 3 models predict churn",
            "Value": int(
                integrated["All_Models_Predict_Churn"].sum()
            ),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    """Run the complete Churn + LTV integration pipeline."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("WEEK 4 DAY 3 - CHURN + LTV MODEL INTEGRATION")
    print("=" * 70)

    integrated = build_integration()

    summary = create_summary(integrated)

    integrated.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        encoding="utf-8",
    )

    print()
    print("INPUT FILE:")
    print(LTV_INPUT)

    print()
    print("CUSTOMERS INTEGRATED:", len(integrated))
    print("UNIQUE CUSTOMER IDS:", integrated["customerID"].nunique())

    print()
    print("OUTPUT FILE:")
    print(OUTPUT_CSV)

    print()
    print("SUMMARY FILE:")
    print(SUMMARY_CSV)

    print()
    print("OUTPUT COLUMNS:")
    for column in integrated.columns:
        print(" -", column)

    print()
    print("FIRST 5 INTEGRATED RECORDS:")
    print(integrated.head().to_string(index=False))

    print()
    print("INTEGRATION SUMMARY:")
    print(summary.to_string(index=False))

    print()
    print("SUCCESS: Churn + LTV integration completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()