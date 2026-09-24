"""
Week 3 - Day 4: Customer Lifetime Value (LTV) Calculation & Prediction Module
Contributor: Rajarsh

Contains:
1. LTVCalculator: Actuarial/financial formulas for Historical, Future, and Total LTV.
2. LTVRegressor: Machine Learning regression models (Ridge, Random Forest, XGBoost) to forecast LTV.
3. CustomerLTVPipeline: End-to-end data processing, LTV modeling, tiering, and reporting.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


class LTVCalculator:
    """
    Mathematical and financial calculations for Customer Lifetime Value (LTV).
    """

    def __init__(self, gross_margin: float = 0.75, discount_rate: float = 0.10):
        self.gross_margin = gross_margin
        self.discount_rate = discount_rate

    def calculate_historical_ltv(self, df: pd.DataFrame) -> pd.Series:
        """
        Historical Realized LTV = Cumulative charges paid to date.
        """
        clean_charges = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        return clean_charges.fillna(df["MonthlyCharges"] * df["tenure"]).round(2)


    def calculate_projected_future_ltv(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates expected future revenue discounted by contract churn hazard.
        - Month-to-month contracts: expected remaining lifetime ~ 10-14 months
        - One-year contracts: expected remaining lifetime ~ 24-30 months
        - Two-year contracts: expected remaining lifetime ~ 48-60 months
        """
        contract_expected_months = {
            "Month-to-month": 12.0,
            "One year": 24.0,
            "Two year": 48.0,
        }
        # Fallback based on Churn status
        expected_months = df["Contract"].map(contract_expected_months).fillna(18.0)
        # If customer has already churned, expected future lifetime is 0
        if "Churn" in df.columns:
            churn_mask = df["Churn"].isin(["Yes", 1])
            expected_months = expected_months.mask(churn_mask, 0.0)

        # Future LTV = MonthlyCharges * Expected Months * Gross Margin
        future_ltv = df["MonthlyCharges"] * expected_months * self.gross_margin
        return future_ltv.round(2)

    def calculate_total_ltv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Historical, Future, and Total Expected LTV, along with LTV Tiers.
        """
        data = df.copy()
        data["Historical_LTV"] = self.calculate_historical_ltv(data)
        data["Future_LTV"] = self.calculate_projected_future_ltv(data)
        data["Total_LTV"] = (data["Historical_LTV"] + data["Future_LTV"]).round(2)

        # Assign LTV Tiers
        data["LTV_Tier"] = self.assign_ltv_tiers(data["Total_LTV"])
        return data

    @staticmethod
    def assign_ltv_tiers(ltv_series: pd.Series) -> pd.Series:
        """
        Segments customers into standard marketing tiers based on LTV percentiles:
        - Platinum: Top 20%
        - Gold: 60th - 80th percentile
        - Silver: 40th - 60th percentile
        - Bronze: Bottom 40%
        """
        percentiles = ltv_series.quantile([0.40, 0.60, 0.80])
        labels = ["Bronze", "Silver", "Gold", "Platinum"]
        bins = [-np.inf, percentiles[0.40], percentiles[0.60], percentiles[0.80], np.inf]
        return pd.cut(ltv_series, bins=bins, labels=labels)


class LTVRegressor:
    """
    Trains and evaluates regression algorithms to predict customer LTV.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
        self.predictions = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names: List[str] = []

    def prepare_features(self, df: pd.DataFrame, target_col: str = "TotalCharges") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses customer demographic, service, and contract attributes for regression.
        """
        data = df.copy()

        # Clean TotalCharges
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"])

        # Target variable (LTV target: TotalCharges / Historical LTV)
        y = data[target_col]

        # Drop identifiers and targets
        drop_cols = ["customerID", "TotalCharges", "Historical_LTV", "Future_LTV", "Total_LTV", "LTV_Tier"]
        drop_cols = [c for c in drop_cols if c in data.columns]
        X = data.drop(columns=drop_cols)

        # Feature transformations
        if "Churn" in X.columns:
            X["Churn"] = X["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0)

        # Binary conversions
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

        if "gender" in X.columns:
            X["is_female"] = (X["gender"] == "Female").astype(int)
            X = X.drop(columns=["gender"])

        # Service count
        service_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        active_service_cols = [c for c in service_cols if c in X.columns]
        if active_service_cols:
            X["service_count"] = (X[active_service_cols] == "Yes").sum(axis=1)

        # One-Hot Encoding
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)

        self.feature_names = X.columns.tolist()
        return X, y

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Trains Ridge, Random Forest, and XGBoost regressors and computes MAE, RMSE, R2, and MAPE.
        """
        print("\n" + "=" * 74)
        print("  WEEK 3 - DAY 4: LTV REGRESSION MODEL BENCHMARK")
        print("=" * 74)

        # Define candidate regression models
        self.models = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "XGBoost Regressor": XGBRegressor(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

        records = []
        best_r2 = -float("inf")

        for name, model in self.models.items():
            print(f"\n[*] Training {name}...")
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)
            self.predictions[name] = y_pred

            mae = mean_absolute_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            self.metrics[name] = {
                "MAE ($)": round(float(mae), 2),
                "RMSE ($)": round(float(rmse), 2),
                "R2 Score": round(float(r2), 4),
                "MAPE (%)": round(float(mape * 100), 2),
            }

            records.append({
                "Model": name,
                "MAE ($)": round(float(mae), 2),
                "RMSE ($)": round(float(rmse), 2),
                "R2 Score": round(float(r2), 4),
                "MAPE (%)": round(float(mape * 100), 2),
            })

            print(f"    - MAE:      ${mae:,.2f}")
            print(f"    - RMSE:     ${rmse:,.2f}")
            print(f"    - R2 Score: {r2:.4f}")
            print(f"    - MAPE:     {mape * 100:.2f}%")

            if r2 > best_r2:
                best_r2 = r2
                self.best_model_name = name
                self.best_model = model

        print(f"\n[+] Champion Model: {self.best_model_name} (R2: {best_r2:.4f})")
        return pd.DataFrame(records)


class CustomerLTVPipeline:
    """
    Coordinates data ingestion, LTV financial calculation, ML regression, and reporting.
    """

    def __init__(self, reports_dir: str = None, models_dir: str = None):
        base_dir = os.path.dirname(__file__)
        self.reports_dir = reports_dir or os.path.join(base_dir, "reports")
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.calculator = LTVCalculator()
        self.regressor = LTVRegressor()

    def run_pipeline(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end LTV pipeline.
        """
        # Step 1: Compute LTV Formulas & Tiers
        print("[*] Calculating Historical, Projected Future, and Total LTV...")
        ltv_df = self.calculator.calculate_total_ltv(df_raw)

        # Step 2: Prepare Features for Regression
        print("[*] Engineering features for LTV regression...")
        X, y = self.regressor.prepare_features(df_raw, target_col="TotalCharges")

        # Step 3: Train-Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        continuous_cols = [c for c in ["tenure", "MonthlyCharges", "service_count"] if c in X_train.columns]
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        if continuous_cols:
            X_train_scaled[continuous_cols] = self.regressor.scaler.fit_transform(X_train[continuous_cols])
            X_test_scaled[continuous_cols] = self.regressor.scaler.transform(X_test[continuous_cols])

        # Step 4: Model Training & Benchmark
        benchmark_df = self.regressor.train_and_evaluate(
            X_train_scaled, X_test_scaled, y_train, y_test
        )

        # Step 5: Save Reports & Visualizations
        self.generate_visualizations(ltv_df, y_test, self.regressor.predictions[self.regressor.best_model_name])
        self.save_artifacts(benchmark_df)

        return ltv_df, benchmark_df

    def generate_visualizations(self, ltv_df: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray):
        """Generates plots for LTV distribution, actual vs predicted, and tier segmentation."""
        plt.style.use("tableau-colorblind10" if "tableau-colorblind10" in plt.style.available else "default")

        # 1. Actual vs Predicted Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.4, color="#1d3557", edgecolors="none", s=25)
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], color="#e63946", lw=2, linestyle="--", label="Perfect Prediction ($y = \\hat{y}$)")
        plt.title(f"LTV Prediction: Actual vs Predicted ({self.regressor.best_model_name})", fontsize=13, fontweight="bold")
        plt.xlabel("Actual Customer LTV ($)", fontsize=11)
        plt.ylabel("Predicted Customer LTV ($)", fontsize=11)
        plt.xlim(0, max_val * 1.05)
        plt.ylim(0, max_val * 1.05)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        act_pred_path = os.path.join(self.reports_dir, "ltv_actual_vs_predicted.png")
        plt.savefig(act_pred_path, dpi=300)
        plt.close()
        print(f"[+] Saved actual vs predicted plot to: {act_pred_path}")

        # 2. LTV Distribution by Contract Type
        plt.figure(figsize=(9, 5.5))
        for contract in ltv_df["Contract"].unique():
            subset = ltv_df[ltv_df["Contract"] == contract]["Total_LTV"]
            plt.hist(subset, bins=30, alpha=0.5, label=f"{contract} (Mean: ${subset.mean():,.0f})")
        plt.title("Customer Total LTV Distribution by Contract Type", fontsize=13, fontweight="bold")
        plt.xlabel("Total Expected LTV ($)", fontsize=11)
        plt.ylabel("Number of Customers", fontsize=11)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        dist_path = os.path.join(self.reports_dir, "ltv_distribution.png")
        plt.savefig(dist_path, dpi=300)
        plt.close()
        print(f"[+] Saved LTV distribution chart to: {dist_path}")

        # 3. LTV Tiers & Value Contribution
        tier_counts = ltv_df["LTV_Tier"].value_counts()[["Bronze", "Silver", "Gold", "Platinum"]]
        tier_revenue = ltv_df.groupby("LTV_Tier", observed=True)["Total_LTV"].sum()[["Bronze", "Silver", "Gold", "Platinum"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        tier_counts.plot(kind="bar", ax=ax1, color=["#b08968", "#90a4ae", "#f4a261", "#2a9d8f"])
        ax1.set_title("Customer Count by LTV Tier", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Customers", fontsize=11)
        ax1.set_xticklabels(["Bronze", "Silver", "Gold", "Platinum"], rotation=0)
        ax1.grid(axis="y", alpha=0.3)

        tier_revenue.plot(kind="bar", ax=ax2, color=["#b08968", "#90a4ae", "#f4a261", "#2a9d8f"])
        ax2.set_title("Total Revenue Contribution ($) by Tier", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Total LTV Revenue ($)", fontsize=11)
        ax2.set_xticklabels(["Bronze", "Silver", "Gold", "Platinum"], rotation=0)
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle("Customer Value Segmentation (LTV Tiers)", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        tier_path = os.path.join(self.reports_dir, "ltv_tier_segmentation.png")
        plt.savefig(tier_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved LTV tier segmentation chart to: {tier_path}")

        # 4. Feature Importance for LTV
        if hasattr(self.regressor.best_model, "feature_importances_"):
            importances = self.regressor.best_model.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": self.regressor.feature_names,
                "Importance": importances,
            }).sort_values(by="Importance", ascending=False).head(10)

            plt.figure(figsize=(9, 5))
            plt.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1], color="#2b5c8f")
            plt.title(f"Top 10 Drivers of Customer LTV ({self.regressor.best_model_name})", fontsize=13, fontweight="bold")
            plt.xlabel("Relative Importance", fontsize=11)
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            feat_path = os.path.join(self.reports_dir, "ltv_feature_importance.png")
            plt.savefig(feat_path, dpi=300)
            plt.close()
            print(f"[+] Saved LTV feature importance chart to: {feat_path}")

    def save_artifacts(self, benchmark_df: pd.DataFrame):
        """Saves model binary, scaler, and metadata."""
        # 1. Model binary
        model_path = os.path.join(self.models_dir, "best_ltv_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.regressor.best_model, f)
        print(f"[+] Saved LTV regression model binary to: {model_path}")

        # 2. Scaler
        scaler_path = os.path.join(self.models_dir, "ltv_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.regressor.scaler, f)

        # 3. Metadata
        metadata = {
            "author": "Rajarsh",
            "task": "Week 3 Day 4: LTV prediction/calculation module and notebook",
            "champion_model": self.regressor.best_model_name,
            "metrics": self.regressor.metrics,
            "feature_names": self.regressor.feature_names,
            "total_features": len(self.regressor.feature_names),
        }
        meta_path = os.path.join(self.models_dir, "ltv_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Saved LTV metadata to: {meta_path}")

        # 4. Evaluation CSV
        csv_path = os.path.join(self.reports_dir, "ltv_evaluation_metrics.csv")
        benchmark_df.to_csv(csv_path, index=False)
        print(f"[+] Saved LTV benchmark metrics to: {csv_path}")
