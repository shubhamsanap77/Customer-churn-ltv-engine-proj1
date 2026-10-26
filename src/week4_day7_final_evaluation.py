from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "week4_day7_final_evaluation"
    / "reference"
)

DAY3_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "week4_day3_integration"
    / "churn_ltv_integrated_customer_output.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "week4_day7_final_evaluation"
)

FINAL_METRICS = OUTPUT_DIR / "final_model_evaluation.csv"
CLASSIFICATION_REPORTS = OUTPUT_DIR / "final_classification_reports.txt"
VALIDATION_SUMMARY = OUTPUT_DIR / "final_validation_summary.csv"
SUMMARY_TEXT = OUTPUT_DIR / "final_evaluation_summary.txt"
COMPARISON_PLOT = OUTPUT_DIR / "final_model_comparison.png"


MODEL_MAPPING = {
    "Logistic Regression": "Logistic_Regression",
    "Decision Tree": "Decision_Tree",
    "Random Forest": "Random_Forest",
}


def validate_reference_files() -> None:
    """Ensure the Week 4 Day 2 reference artifacts exist."""

    required_files = [
        REFERENCE_DIR / "week4_day2_model_comparison.csv",
        REFERENCE_DIR / "week4_day2_predictions.csv",
    ]

    missing = [str(path) for path in required_files if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing Week 4 Day 2 reference file(s):\n"
            + "\n".join(missing)
        )


def evaluate_reference_predictions() -> tuple[pd.DataFrame, str]:
    """
    Independently calculate hard-classification metrics from the
    Week 4 Day 2 prediction output and compare them with the
    reported reference metrics.
    """

    prediction_path = REFERENCE_DIR / "week4_day2_predictions.csv"
    comparison_path = REFERENCE_DIR / "week4_day2_model_comparison.csv"

    predictions = pd.read_csv(prediction_path)
    reference = pd.read_csv(comparison_path)

    if list(predictions.columns) != [
        "Actual",
        "Logistic_Regression",
        "Decision_Tree",
        "Random_Forest",
    ]:
        raise ValueError(
            f"Unexpected prediction columns: {list(predictions.columns)}"
        )

    if len(predictions) != 1409:
        raise ValueError(
            f"Expected 1409 Week 4 Day 2 test rows, found {len(predictions)}."
        )

    actual_counts = predictions["Actual"].value_counts().to_dict()

    if actual_counts.get(0, 0) != 1035:
        raise ValueError(
            f"Expected 1035 non-churn test records, found "
            f"{actual_counts.get(0, 0)}."
        )

    if actual_counts.get(1, 0) != 374:
        raise ValueError(
            f"Expected 374 churn test records, found "
            f"{actual_counts.get(1, 0)}."
        )

    results = []

    for model_name, prediction_column in MODEL_MAPPING.items():

        y_true = predictions["Actual"]
        y_pred = predictions[prediction_column]

        calculated = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "F1 Score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }

        reference_row = reference[
            reference["Model"] == model_name
        ]

        if reference_row.empty:
            raise ValueError(
                f"No reference metrics found for {model_name}."
            )

        reference_row = reference_row.iloc[0]

        hard_metric_checks = {
            "Accuracy": np.isclose(
                calculated["Accuracy"],
                reference_row["Accuracy"],
                atol=1e-9,
            ),
            "Precision": np.isclose(
                calculated["Precision"],
                reference_row["Precision"],
                atol=1e-9,
            ),
            "Recall": np.isclose(
                calculated["Recall"],
                reference_row["Recall"],
                atol=1e-9,
            ),
            "F1 Score": np.isclose(
                calculated["F1 Score"],
                reference_row["F1 Score"],
                atol=1e-9,
            ),
        }

        calculated["ROC-AUC"] = reference_row["ROC-AUC"]

        calculated["Accuracy_Verified"] = hard_metric_checks["Accuracy"]
        calculated["Precision_Verified"] = hard_metric_checks["Precision"]
        calculated["Recall_Verified"] = hard_metric_checks["Recall"]
        calculated["F1_Verified"] = hard_metric_checks["F1 Score"]

        calculated["Hard_Metrics_Verified"] = all(
            hard_metric_checks.values()
        )

        calculated["ROC_AUC_Source"] = (
            "Week 4 Day 2 reported reference result"
        )

        results.append(calculated)

    metrics = pd.DataFrame(results)

    if not metrics["Hard_Metrics_Verified"].all():
        raise ValueError(
            "At least one independently recalculated metric does not "
            "match the Week 4 Day 2 reference."
        )

    reports = []

    for model_name, prediction_column in MODEL_MAPPING.items():

        reports.append(
            f"{model_name}\n"
            + "-" * len(model_name)
            + "\n"
            + classification_report(
                predictions["Actual"],
                predictions[prediction_column],
                digits=4,
                zero_division=0,
            )
            + "\n"
        )

    return metrics, "\n".join(reports)


def validate_day3_integration() -> pd.DataFrame:
    """Validate the actual Week 4 Day 3 Churn + LTV output."""

    if not DAY3_OUTPUT.exists():
        raise FileNotFoundError(
            f"Week 4 Day 3 integration output not found: {DAY3_OUTPUT}"
        )

    df = pd.read_csv(DAY3_OUTPUT)

    required_columns = [
        "customerID",
        "Actual_Churn",
        "Estimated_LTV",
        "Logistic_Churn_Prediction",
        "Logistic_Churn_Probability",
        "DecisionTree_Churn_Prediction",
        "DecisionTree_Churn_Probability",
        "RandomForest_Churn_Prediction",
        "RandomForest_Churn_Probability",
        "Churn_Model_Agreement_Count",
        "All_Models_Predict_Churn",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Week 4 Day 3 output is missing columns: {missing_columns}"
        )

    validation_rows = []

    def add_check(name: str, passed: bool, value) -> None:
        validation_rows.append(
            {
                "Validation": name,
                "Status": "PASS" if passed else "FAIL",
                "Value": value,
            }
        )

    add_check(
        "Integrated row count",
        len(df) == 5174,
        len(df),
    )

    add_check(
        "Unique customer IDs",
        df["customerID"].nunique() == len(df),
        df["customerID"].nunique(),
    )

    add_check(
        "Missing customer IDs",
        df["customerID"].isna().sum() == 0,
        int(df["customerID"].isna().sum()),
    )

    add_check(
        "Missing Estimated_LTV",
        df["Estimated_LTV"].isna().sum() == 0,
        int(df["Estimated_LTV"].isna().sum()),
    )

    add_check(
        "Estimated_LTV numeric",
        pd.api.types.is_numeric_dtype(df["Estimated_LTV"]),
        str(df["Estimated_LTV"].dtype),
    )

    add_check(
        "Estimated_LTV minimum non-negative",
        df["Estimated_LTV"].min() >= 0,
        float(df["Estimated_LTV"].min()),
    )

    prediction_columns = [
        "Logistic_Churn_Prediction",
        "DecisionTree_Churn_Prediction",
        "RandomForest_Churn_Prediction",
    ]

    probability_columns = [
        "Logistic_Churn_Probability",
        "DecisionTree_Churn_Probability",
        "RandomForest_Churn_Probability",
    ]

    for column in prediction_columns:
        valid = df[column].dropna().isin([0, 1]).all()

        add_check(
            f"{column} contains only 0/1",
            valid,
            sorted(df[column].dropna().unique().tolist()),
        )

    for column in probability_columns:

        valid_numeric = pd.api.types.is_numeric_dtype(df[column])
        valid_range = (
            df[column].between(0, 1).all()
            if valid_numeric
            else False
        )

        add_check(
            f"{column} in [0, 1]",
            valid_numeric and valid_range,
            (
                float(df[column].min()),
                float(df[column].max()),
            )
            if valid_numeric
            else "non-numeric",
        )

    add_check(
        "No missing values in integrated output",
        df.isna().sum().sum() == 0,
        int(df.isna().sum().sum()),
    )

    validation = pd.DataFrame(validation_rows)

    if not validation["Status"].eq("PASS").all():
        failed = validation[
            validation["Status"] == "FAIL"
        ]

        raise ValueError(
            "Week 4 Day 3 validation failure:\n"
            + failed.to_string(index=False)
        )

    return validation


def create_comparison_plot(metrics: pd.DataFrame) -> None:
    """Create a final model-comparison chart."""

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ]

    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 7))

    for index, metric in enumerate(metric_columns):
        ax.bar(
            x + (index - 2) * width,
            metrics[metric],
            width,
            label=metric,
        )

    ax.set_title("Week 4 Final Churn Model Evaluation")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["Model"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        COMPARISON_PLOT,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write a human-readable final evaluation summary."""

    best_f1_model = metrics.loc[
        metrics["F1 Score"].idxmax(),
        "Model",
    ]

    best_roc_model = metrics.loc[
        metrics["ROC-AUC"].idxmax(),
        "Model",
    ]

    all_hard_metrics_verified = bool(
        metrics["Hard_Metrics_Verified"].all()
    )

    all_integration_checks_passed = bool(
        validation["Status"].eq("PASS").all()
    )

    status = (
        all_hard_metrics_verified
        and all_integration_checks_passed
    )

    lines = [
        "WEEK 4 DAY 7 - FINAL MODEL EVALUATION SUMMARY",
        "=" * 60,
        "",
        "FINAL STATUS:",
        "PASS" if status else "FAIL",
        "",
        "WEEK 4 DAY 2 EVALUATION REFERENCE",
        f"Test rows: 1409",
        "Actual class distribution: 1035 non-churn / 374 churn",
        "",
        "MODEL RESULTS:",
        metrics[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ]
        ].to_string(index=False),
        "",
        f"Highest F1 Score: {best_f1_model}",
        f"Highest ROC-AUC: {best_roc_model}",
        "",
        "VERIFICATION NOTE:",
        "Accuracy, Precision, Recall and F1 were independently "
        "recalculated from the Week 4 Day 2 prediction file and "
        "matched the reported reference values exactly.",
        "",
        "ROC-AUC NOTE:",
        "ROC-AUC values are preserved from the Week 4 Day 2 "
        "reference metrics because the available prediction file "
        "contains hard class predictions rather than probability/"
        "decision scores required to independently recalculate "
        "ROC-AUC.",
        "",
        "WEEK 4 DAY 3 INTEGRATION VALIDATION:",
        validation.to_string(index=False),
        "",
        "OUTPUTS:",
        str(FINAL_METRICS),
        str(CLASSIFICATION_REPORTS),
        str(VALIDATION_SUMMARY),
        str(SUMMARY_TEXT),
        str(COMPARISON_PLOT),
        "",
        "SUCCESS: Week 4 final evaluation completed.",
    ]

    SUMMARY_TEXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("WEEK 4 DAY 7 - FINAL MODEL EVALUATION")
    print("=" * 70)

    validate_reference_files()

    print()
    print("1. Verifying Week 4 Day 2 reference metrics...")
    metrics, reports = evaluate_reference_predictions()

    print("   HARD METRICS VERIFIED: PASS")

    print()
    print("2. Validating Week 4 Day 3 integrated output...")
    validation = validate_day3_integration()

    print("   INTEGRATION VALIDATION: PASS")

    metrics.to_csv(
        FINAL_METRICS,
        index=False,
        encoding="utf-8",
    )

    CLASSIFICATION_REPORTS.write_text(
        reports,
        encoding="utf-8",
    )

    validation.to_csv(
        VALIDATION_SUMMARY,
        index=False,
        encoding="utf-8",
    )

    create_comparison_plot(metrics)

    write_summary(
        metrics,
        validation,
    )

    print()
    print("3. OUTPUT FILES CREATED:")
    print(" -", FINAL_METRICS)
    print(" -", CLASSIFICATION_REPORTS)
    print(" -", VALIDATION_SUMMARY)
    print(" -", SUMMARY_TEXT)
    print(" -", COMPARISON_PLOT)

    print()
    print("4. FINAL MODEL RESULTS:")
    print(
        metrics[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ]
        ].to_string(index=False)
    )

    print()
    print("SUCCESS: Week 4 Day 7 final evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()