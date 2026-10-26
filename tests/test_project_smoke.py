from pathlib import Path
import unittest
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_DIR = (
    ROOT / "data" / "processed" / "week3_day3_evaluation"
)


class ProjectSmokeTests(unittest.TestCase):

    def test_source_scripts_exist(self):
        files = [
            ROOT / "src" / "day3_feature_preparation.py",
            ROOT / "src" / "model_comparison.py",
            ROOT / "src" / "week3_day3_model_evaluation.py",
        ]

        for file_path in files:
            self.assertTrue(
                file_path.exists(),
                f"Missing source file: {file_path}"
            )

    def test_evaluation_outputs_exist(self):
        files = [
            "classification_reports.txt",
            "confusion_matrices.json",
            "evaluation_summary.txt",
            "model_comparison.png",
            "model_evaluation_metrics.csv",
        ]

        for filename in files:
            self.assertTrue(
                (EVALUATION_DIR / filename).exists(),
                f"Missing output file: {filename}"
            )

    def test_metrics_file_structure(self):
        metrics_file = EVALUATION_DIR / "model_evaluation_metrics.csv"

        self.assertTrue(metrics_file.exists())

        df = pd.read_csv(metrics_file)

        expected_columns = {
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        }

        self.assertTrue(
            expected_columns.issubset(df.columns),
            f"Missing columns: {expected_columns - set(df.columns)}"
        )

        self.assertEqual(len(df), 3)

        self.assertEqual(
            set(df["model"]),
            {
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
            },
        )

    def test_metric_values_are_valid(self):
        metrics_file = EVALUATION_DIR / "model_evaluation_metrics.csv"

        df = pd.read_csv(metrics_file)

        metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]

        for column in metrics:
            self.assertTrue(df[column].notna().all())
            self.assertTrue((df[column] >= 0).all())
            self.assertTrue((df[column] <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)