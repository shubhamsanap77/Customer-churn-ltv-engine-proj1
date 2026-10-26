from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = (
    ROOT
    / "data"
    / "processed"
    / "week4_day7_final_evaluation"
)

INTEGRATION_DIR = (
    ROOT
    / "data"
    / "processed"
    / "week4_day3_integration"
)

FINAL_METRICS = FINAL_DIR / "final_model_evaluation.csv"
FINAL_VALIDATION = FINAL_DIR / "final_validation_summary.csv"
FINAL_SUMMARY = FINAL_DIR / "final_evaluation_summary.txt"
FINAL_REPORTS = FINAL_DIR / "final_classification_reports.txt"
FINAL_PLOT = FINAL_DIR / "final_model_comparison.png"

INTEGRATED_OUTPUT = (
    INTEGRATION_DIR
    / "churn_ltv_integrated_customer_output.csv"
)


class Week4Day7FinalTests(unittest.TestCase):

    def test_final_evaluation_script_exists(self):
        script = ROOT / "src" / "week4_day7_final_evaluation.py"

        self.assertTrue(
            script.exists(),
            f"Missing final evaluation script: {script}"
        )

    def test_final_evaluation_outputs_exist(self):
        required_files = [
            FINAL_METRICS,
            FINAL_VALIDATION,
            FINAL_SUMMARY,
            FINAL_REPORTS,
            FINAL_PLOT,
        ]

        for file_path in required_files:
            self.assertTrue(
                file_path.exists(),
                f"Missing final evaluation output: {file_path}"
            )

    def test_final_metrics_structure(self):
        self.assertTrue(FINAL_METRICS.exists())

        df = pd.read_csv(FINAL_METRICS)

        expected_columns = {
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Accuracy_Verified",
            "Precision_Verified",
            "Recall_Verified",
            "F1_Verified",
            "Hard_Metrics_Verified",
            "ROC_AUC_Source",
        }

        self.assertTrue(
            expected_columns.issubset(df.columns),
            f"Missing columns: "
            f"{expected_columns - set(df.columns)}"
        )

        self.assertEqual(len(df), 3)

        self.assertEqual(
            set(df["Model"]),
            {
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            },
        )

    def test_final_metrics_are_valid(self):
        df = pd.read_csv(FINAL_METRICS)

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
        ]

        for column in metric_columns:
            self.assertTrue(
                df[column].notna().all(),
                f"{column} contains missing values"
            )

            self.assertTrue(
                df[column].between(0, 1).all(),
                f"{column} contains values outside [0, 1]"
            )

        self.assertTrue(
            df["Accuracy_Verified"].all()
        )

        self.assertTrue(
            df["Precision_Verified"].all()
        )

        self.assertTrue(
            df["Recall_Verified"].all()
        )

        self.assertTrue(
            df["F1_Verified"].all()
        )

        self.assertTrue(
            df["Hard_Metrics_Verified"].all()
        )

    def test_integration_output(self):
        self.assertTrue(INTEGRATED_OUTPUT.exists())

        df = pd.read_csv(INTEGRATED_OUTPUT)

        self.assertEqual(len(df), 5174)

        self.assertEqual(
            df["customerID"].nunique(),
            5174,
        )

        self.assertEqual(
            df["customerID"].isna().sum(),
            0,
        )

        self.assertEqual(
            df["Estimated_LTV"].isna().sum(),
            0,
        )

        probability_columns = [
            "Logistic_Churn_Probability",
            "DecisionTree_Churn_Probability",
            "RandomForest_Churn_Probability",
        ]

        for column in probability_columns:
            self.assertTrue(
                df[column].between(0, 1).all(),
                f"{column} contains invalid probabilities"
            )

        prediction_columns = [
            "Logistic_Churn_Prediction",
            "DecisionTree_Churn_Prediction",
            "RandomForest_Churn_Prediction",
        ]

        for column in prediction_columns:
            self.assertTrue(
                df[column].isin([0, 1]).all(),
                f"{column} contains values other than 0/1"
            )

    def test_validation_summary_passes(self):
        self.assertTrue(FINAL_VALIDATION.exists())

        df = pd.read_csv(FINAL_VALIDATION)

        self.assertTrue(
            (df["Status"] == "PASS").all(),
            "One or more final validation checks failed"
        )

        self.assertEqual(
            int(
                df.loc[
                    df["Validation"] == "Integrated row count",
                    "Value",
                ].iloc[0]
            ),
            5174,
        )

    def test_final_summary_reports_pass(self):
        self.assertTrue(FINAL_SUMMARY.exists())

        text = FINAL_SUMMARY.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "FINAL STATUS:",
            text
        )

        self.assertIn(
            "PASS",
            text
        )

        self.assertIn(
            "SUCCESS: Week 4 final evaluation completed.",
            text
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)