"""
Week 2: Automated Pipeline & Data Integrity Tests
Validates dataset hygiene, feature engineering logic, leakage prevention,
scaler calibration, and model prediction bounds.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from week_2.data_loader import load_raw_data
    from week_2.feature_engineering import FeatureEngineer
except ImportError:
    from data_loader import load_raw_data
    from feature_engineering import FeatureEngineer


class TestWeek2Pipeline(unittest.TestCase):
    """
    Test suite for Week 2 data preprocessing, feature engineering, and model integrity.
    """

    @classmethod
    def setUpClass(cls):
        """Loads data once for the test suite."""
        cls.df_raw = load_raw_data()
        cls.fe = FeatureEngineer()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.features = cls.fe.fit_transform(cls.df_raw)

    def test_01_raw_data_shape(self):
        """Validates that the raw Telco dataset meets expected row and column constraints."""
        self.assertGreaterEqual(self.df_raw.shape[0], 7000, "Dataset should have over 7,000 rows")
        self.assertGreaterEqual(self.df_raw.shape[1], 20, "Dataset should have at least 20 columns")
        self.assertIn("Churn", self.df_raw.columns, "Target column 'Churn' must exist")

    def test_02_total_charges_cleanup(self):
        """Ensures TotalCharges whitespace strings are converted to valid floats with zero nulls."""
        cleaned = self.fe.clean_raw_data(self.df_raw)
        self.assertTrue(pd.api.types.is_float_dtype(cleaned["TotalCharges"]))
        self.assertEqual(cleaned["TotalCharges"].isnull().sum(), 0, "No nulls allowed in TotalCharges after cleaning")

    def test_03_no_target_leakage(self):
        """Asserts that target column 'Churn' and identifier 'customerID' are excluded from features."""
        self.assertNotIn("Churn", self.X_train.columns, "Leakage alert: 'Churn' found in training features!")
        self.assertNotIn("customerID", self.X_train.columns, "Leakage alert: 'customerID' found in training features!")
        self.assertNotIn("Churn", self.X_test.columns, "Leakage alert: 'Churn' found in test features!")

    def test_04_engineered_features_present(self):
        """Verifies presence of key domain-engineered features."""
        expected_features = [
            "bill_shock",
            "monthly_to_total_ratio",
            "is_month_to_month",
            "service_bundle_count",
            "has_security_bundle",
            "is_electronic_check",
        ]
        for feat in expected_features:
            self.assertIn(feat, self.features, f"Engineered feature '{feat}' is missing from feature pipeline")

    def test_05_train_test_split_stratification(self):
        """Verifies stratified split preserves churn ratio within 1% margin."""
        train_churn_rate = self.y_train.mean()
        test_churn_rate = self.y_test.mean()
        self.assertAlmostEqual(
            train_churn_rate,
            test_churn_rate,
            delta=0.015,
            msg="Train and test churn rates should be closely stratified",
        )

    def test_06_scaler_calibration(self):
        """Validates that continuous scaled features on train set have mean ~ 0 and std ~ 1."""
        for num_col in ["tenure", "MonthlyCharges", "bill_shock"]:
            if num_col in self.X_train.columns:
                mean_val = self.X_train[num_col].mean()
                std_val = self.X_train[num_col].std()
                self.assertAlmostEqual(mean_val, 0.0, delta=0.1, msg=f"{num_col} scaled mean should be near 0")
                self.assertAlmostEqual(std_val, 1.0, delta=0.1, msg=f"{num_col} scaled std should be near 1")

    def test_07_prediction_shapes_and_probabilities(self):
        """Tests that classifier predictions match test size and probabilities fall in [0, 1]."""
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(self.X_train, self.y_train)

        preds = clf.predict(self.X_test)
        probs = clf.predict_proba(self.X_test)[:, 1]

        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be within [0, 1]")
        self.assertTrue(set(preds).issubset({0, 1}), "Predictions must be binary {0, 1}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RUNNING WEEK 2 PIPELINE INTEGRITY & UNIT TESTS")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWeek2Pipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[+] All 7 pipeline integrity tests passed successfully!")
    else:
        sys.exit(1)
