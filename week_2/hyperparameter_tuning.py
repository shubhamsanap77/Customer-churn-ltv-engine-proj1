"""
Week 2: Hyperparameter Tuning & Cross-Validation Module
Performs 5-fold Stratified K-Fold tuning for Random Forest and XGBoost classifiers.
Optimizes for F1-Score to maximize customer churn detection capability.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from week_2.data_loader import load_raw_data
    from week_2.feature_engineering import FeatureEngineer
except ImportError:
    from data_loader import load_raw_data
    from feature_engineering import FeatureEngineer


class HyperparameterTuner:
    """
    Performs grid search and stratified cross-validation to find optimal model hyperparameters.
    """

    def __init__(self, reports_dir: str = None, models_dir: str = None):
        base_dir = os.path.dirname(__file__)
        self.reports_dir = reports_dir or os.path.join(base_dir, "reports")
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.results = {}
        self.best_tuned_model = None
        self.best_tuned_name = None

    def tune_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Executes cross-validated grid search over Random Forest and XGBoost.
        """
        print("\n" + "=" * 70)
        print("  WEEK 2: HYPERPARAMETER TUNING (5-FOLD STRATIFIED CV)")
        print("=" * 70)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

        # Parameter Grids
        rf_grid = {
            "n_estimators": [150, 250],
            "max_depth": [6, 10],
            "min_samples_leaf": [2, 4],
            "class_weight": ["balanced"],
        }

        xgb_grid = {
            "n_estimators": [150, 250],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.08],
            "scale_pos_weight": [pos_weight],
            "subsample": [0.8, 1.0],
            "eval_metric": ["logloss"],
        }

        models_to_tune = {
            "Random Forest (Tuned)": (
                RandomForestClassifier(random_state=42, n_jobs=-1),
                rf_grid,
            ),
            "XGBoost (Tuned)": (
                XGBClassifier(random_state=42, n_jobs=-1),
                xgb_grid,
            ),
        }

        records = []
        best_f1 = -1.0

        for name, (base_model, param_grid) in models_to_tune.items():
            print(f"\n[*] Tuning {name} across {len(param_grid)} parameter dimensions...")
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring="f1",
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            grid_search.fit(X_train, y_train)

            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)
            y_prob = (
                best_estimator.predict_proba(X_test)[:, 1]
                if hasattr(best_estimator, "predict_proba")
                else y_pred
            )

            test_f1 = f1_score(y_test, y_pred, zero_division=0)
            test_auc = roc_auc_score(y_test, y_prob)
            test_rec = recall_score(y_test, y_pred, zero_division=0)
            test_prec = precision_score(y_test, y_pred, zero_division=0)

            print(f"    - Best CV F1 Score:  {grid_search.best_score_:.4f}")
            print(f"    - Test Set F1 Score: {test_f1:.4f}")
            print(f"    - Test Set Recall:   {test_rec:.4f}")
            print(f"    - Test Set ROC-AUC:  {test_auc:.4f}")
            print(f"    - Best Parameters:   {grid_search.best_params_}")

            records.append({
                "Model": name,
                "Best_CV_F1": round(float(grid_search.best_score_), 4),
                "Test_F1": round(float(test_f1), 4),
                "Test_Recall": round(float(test_rec), 4),
                "Test_Precision": round(float(test_prec), 4),
                "Test_ROC_AUC": round(float(test_auc), 4),
                "Best_Params": str(grid_search.best_params_),
            })

            if test_f1 > best_f1:
                best_f1 = test_f1
                self.best_tuned_name = name
                self.best_tuned_model = best_estimator

        results_df = pd.DataFrame(records)
        csv_path = os.path.join(self.reports_dir, "tuning_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[+] Saved tuning results to: {csv_path}")

        self._plot_tuning_performance(results_df)
        self._save_best_model()

        return results_df

    def _plot_tuning_performance(self, results_df: pd.DataFrame):
        """Plots CV vs Test performance for tuned models."""
        plt.figure(figsize=(9, 5))
        x = np.arange(len(results_df))
        width = 0.35

        plt.bar(x - width / 2, results_df["Best_CV_F1"], width, label="CV F1-Score", color="#2b5c8f")
        plt.bar(x + width / 2, results_df["Test_F1"], width, label="Test F1-Score", color="#e76f51")

        plt.xticks(x, results_df["Model"], fontsize=11, fontweight="semibold")
        plt.ylabel("F1-Score", fontsize=11)
        plt.ylim(0, 1.0)
        plt.title("5-Fold Cross-Validation vs Test F1 Performance", fontsize=13, fontweight="bold")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, "cv_hyperparameter_performance.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved tuning chart to: {plot_path}")

    def _save_best_model(self):
        """Serializes the best tuned model if available."""
        if self.best_tuned_model is not None:
            model_path = os.path.join(self.models_dir, "tuned_churn_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.best_tuned_model, f)
            print(f"[+] Saved best tuned model ({self.best_tuned_name}) to: {model_path}")


if __name__ == "__main__":
    df = load_raw_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)

    tuner = HyperparameterTuner()
    summary = tuner.tune_models(X_train, y_train, X_test, y_test)
    print("\n[+] Hyperparameter Tuning Summary:")
    print(summary[["Model", "Best_CV_F1", "Test_F1", "Test_Recall", "Test_ROC_AUC"]])
