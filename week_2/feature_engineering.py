"""
Week 2: Feature Engineering & Preprocessing Pipeline
Prepares cleaned, scaled numerical and encoded categorical features for model training.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    Handles preprocessing, feature transformations, scaling, and train-test splitting.
    """

    def __init__(self, target_col: str = "Churn", id_col: str = "customerID", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.id_col = id_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans missing/blank values and formats types."""
        data = df.copy()

        # Handle whitespace in TotalCharges
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"])

        # Map binary target Churn
        if self.target_col in data.columns:
            data[self.target_col] = data[self.target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

        return data

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs domain-specific churn indicators and encodes variables."""
        data = self.clean_raw_data(df)

        # 1. Tenure Cohorts
        data["tenure_cohort"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 60, 100],
            labels=["0-12m", "12-24m", "24-48m", "48-60m", "60m+"],
        ).astype(str)

        # 2. Financial Dynamics & Ratios
        data["monthly_to_total_ratio"] = data["MonthlyCharges"] / (data["TotalCharges"] + 1.0)
        data["avg_historical_monthly"] = data["TotalCharges"] / (data["tenure"] + 1.0)
        data["bill_shock"] = data["MonthlyCharges"] - data["avg_historical_monthly"]

        # 3. Service Bundling & Risk Indicators
        service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        data["service_bundle_count"] = (data[service_cols] == "Yes").sum(axis=1)
        data["is_month_to_month"] = (data["Contract"] == "Month-to-month").astype(int)
        data["is_electronic_check"] = (data["PaymentMethod"] == "Electronic check").astype(int)
        data["has_family"] = ((data["Partner"] == "Yes") | (data["Dependents"] == "Yes")).astype(int)

        # Binary conversions
        binary_map = {"Yes": 1, "No": 0}
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in data.columns:
                data[col] = data[col].map(binary_map).fillna(0).astype(int)

        if "gender" in data.columns:
            data["is_female"] = (data["gender"] == "Female").astype(int)

        # Drop ID and raw text
        drop_cols = [self.id_col, "gender"]
        drop_cols = [c for c in drop_cols if c in data.columns]
        data = data.drop(columns=drop_cols)

        # One-hot encode categoricals
        categorical_cols = data.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        if self.target_col in categorical_cols:
            categorical_cols.remove(self.target_col)

        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
        return data

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """Fits preprocessing, scales numerical features, and performs stratified split."""
        featured_df = self.engineer_features(df)

        X = featured_df.drop(columns=[self.target_col])
        y = featured_df[self.target_col]

        self.feature_names = X.columns.tolist()
        numerical_cols = [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "monthly_to_total_ratio",
            "avg_historical_monthly",
            "bill_shock",
            "service_bundle_count",
        ]
        numerical_cols = [c for c in numerical_cols if c in self.feature_names]

        # Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])

        return X_train_scaled, X_test_scaled, y_train, y_test, self.feature_names
