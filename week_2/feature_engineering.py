"""
Week 2: Feature Engineering & Preprocessing Pipeline
Transforms raw Telco customer data into high-value engineered features for churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Feature engineering and data transformation pipeline for Customer Churn.
    """

    def __init__(self, target_col: str = "Churn", id_col: str = "customerID", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.id_col = id_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.numerical_cols: List[str] = []

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans missing values and converts datatypes."""
        data = df.copy()

        # Handle whitespace in TotalCharges
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For new customers with tenure = 0, TotalCharges is 0
        data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"])

        # Encode target variable if present
        if self.target_col in data.columns:
            data[self.target_col] = data[self.target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

        return data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates domain-specific behavioral, financial, and product bundling features.
        """
        data = self.clean_raw_data(df)

        # 1. Tenure Cohorts / Lifecyle Bins
        data["tenure_cohort"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 60, 100],
            labels=["0-12m", "12-24m", "24-48m", "48-60m", "60m+"],
        ).astype(str)

        # 2. Financial & Spend Dynamics
        # Ratio of monthly bill to cumulative spend
        data["monthly_to_total_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1.0)

        # Average historical monthly charges vs current monthly charges
        data["avg_historical_monthly"] = data["TotalCharges"] / (data["tenure"] + 1.0)
        # Bill shock: Is current monthly charge higher than historical average?
        data["bill_shock"] = data["MonthlyCharges"] - data["avg_historical_monthly"]

        # 3. Product Bundling & Engagement Index
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        data["service_bundle_count"] = (data[service_cols] == "Yes").sum(axis=1)

        # High-retention security bundle (TechSupport + OnlineSecurity)
        data["has_security_bundle"] = (
            (data["OnlineSecurity"] == "Yes") & (data["TechSupport"] == "Yes")
        ).astype(int)

        # Entertainment bundle
        data["has_streaming_bundle"] = (
            (data["StreamingTV"] == "Yes") | (data["StreamingMovies"] == "Yes")
        ).astype(int)

        # 4. Service Friction & Churn Risk Indicators
        # Fiber Optic without TechSupport (known high churn cohort in telco)
        data["fiber_without_support"] = (
            (data["InternetService"] == "Fiber optic") & (data["TechSupport"] != "Yes")
        ).astype(int)

        # Contract Risk: Month-to-month contracts have high cancellation propensity
        data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(int)
        data["is_long_term_contract"] = (data["Contract"].isin(["One year", "Two year"])).astype(int)

        # 5. Payment Risk Indicators
        # Electronic check customers have historically higher churn than automated payers
        data["is_electronic_check"] = (data["PaymentMethod"] == "Electronic check").astype(int)
        data["is_automatic_payment"] = (
            data["PaymentMethod"].str.contains("automatic", case=False, na=False)
        ).astype(int)

        # 6. Demographics & Support Structure
        data["has_family"] = ((data["Partner"] == "Yes") | (data["Dependents"] == "Yes")).astype(int)
        data["is_senior_alone"] = (
            (data["SeniorCitizen"] == 1) & (data["Partner"] == "No") & (data["Dependents"] == "No")
        ).astype(int)

        # Standard binary flags for yes/no columns
        binary_map = {"Yes": 1, "No": 0}
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in data.columns:
                data[col] = data[col].map(binary_map).fillna(0).astype(int)

        if "gender" in data.columns:
            data["is_female"] = (data["gender"] == "Female").astype(int)

        # Drop ID and redundant raw categorical columns
        drop_cols = [self.id_col, "gender"]
        drop_cols = [c for c in drop_cols if c in data.columns]
        data = data.drop(columns=drop_cols)

        # 7. One-Hot Encoding for remaining categorical features
        categorical_cols = data.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)

        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)

        return data

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Fits transformations, scales numerical features, and performs stratified train/test split.
        """
        featured_df = self.engineer_features(df)

        X = featured_df.drop(columns=[self.target_col])
        y = featured_df[self.target_col]

        self.feature_names = X.columns.tolist()
        self.numerical_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "monthly_to_total_ratio",
            "avg_historical_monthly",
            "bill_shock",
            "service_bundle_count",
        ]
        self.numerical_cols = [c for c in self.numerical_cols if c in self.feature_names]

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        # Scale continuous features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[self.numerical_cols] = self.scaler.fit_transform(X_train[self.numerical_cols])
        X_test_scaled[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])

        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    try:
        from week_2.data_loader import load_raw_data
    except ImportError:
        from data_loader import load_raw_data
    df = load_raw_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)
    print(f"[+] Features Engineered: {len(features)}")
    print(f"[+] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[+] Churn rate in train: {y_train.mean():.2%}, in test: {y_test.mean():.2%}")
    print(f"[+] Engineered columns sample: {features[:10]}")

