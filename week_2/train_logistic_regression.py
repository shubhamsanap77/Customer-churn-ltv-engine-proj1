"""
Week 2 - Day 4: Logistic Regression Training & Evaluation
Task Assigned to: Rajarsh
Focus: Train Logistic Regression models and evaluate Precision, Recall, F1-Score, and ROC-AUC.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from week_2.data_loader import load_raw_data
    from week_2.feature_engineering import FeatureEngineer
except ImportError:
    from data_loader import load_raw_data
    from feature_engineering import FeatureEngineer


class LogisticRegressionPipeline:
    """
    Manages Logistic Regression model training, evaluation, comparison, and reporting.
    """

    def __init__(self, reports_dir: str = None, models_dir: str = None):
        base_dir = os.path.dirname(__file__)
        self.reports_dir = reports_dir or os.path.join(base_dir, "reports")
        self.models_dir = models_dir or os.path.join(base_dir, "models")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.models = {}
        self.metrics = {}
        self.predictions = {}
        self.probabilities = {}

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Trains Standard and Class-Balanced Logistic Regression classifiers.
        Evaluates Precision, Recall, F1-Score, Accuracy, and ROC-AUC.
        """
        print("\n" + "=" * 72)
        print("  DAY 4: LOGISTIC REGRESSION TRAINING & EVALUATION (RAJARSH)")
        print("=" * 72)

        # 1. Define Model Variants
        self.models = {
            "Logistic Regression (Standard)": LogisticRegression(
                class_weight=None,
                C=1.0,
                max_iter=1000,
                random_state=42,
            ),
            "Logistic Regression (Balanced)": LogisticRegression(
                class_weight="balanced",
                C=0.1,
                max_iter=1000,
                random_state=42,
            ),
        }

        records = []

        for name, model in self.models.items():
            print(f"\n[*] Training {name}...")
            model.fit(X_train, y_train)

            # Generate predictions and probability scores
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            self.predictions[name] = y_pred
            self.probabilities[name] = y_prob

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)

            self.metrics[name] = {
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1-Score": round(float(f1), 4),
                "ROC-AUC": round(float(auc), 4),
            }

            records.append({
                "Model": name,
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1-Score": round(float(f1), 4),
                "Accuracy": round(float(acc), 4),
                "ROC-AUC": round(float(auc), 4),
            })

            print(f"    - Precision: {prec:.4f}")
            print(f"    - Recall:    {rec:.4f}")
            print(f"    - F1-Score:  {f1:.4f}")
            print(f"    - Accuracy:  {acc:.4f}")
            print(f"    - ROC-AUC:   {auc:.4f}")

        results_df = pd.DataFrame(records)

        # Print Detailed Classification Report for the Balanced model (Recommended for Churn)
        best_name = "Logistic Regression (Balanced)"
        print("\n" + "-" * 72)
        print(f"  DETAILED CLASSIFICATION REPORT: {best_name}")
        print("-" * 72)
        print(classification_report(y_test, self.predictions[best_name], target_names=["Retained (0)", "Churned (1)"]))

        # Save results to CSV
        csv_path = os.path.join(self.reports_dir, "logistic_regression_evaluation.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"[+] Saved evaluation metrics to: {csv_path}")

        return results_df

    def generate_plots(self, y_test: pd.Series):
        """Generates and saves visual reports for Logistic Regression evaluation."""
        plt.style.use("tableau-colorblind10" if "tableau-colorblind10" in plt.style.available else "default")

        # 1. Bar Chart Comparison of Precision, Recall, F1
        metric_df = pd.DataFrame(self.metrics).T
        fig, ax = plt.subplots(figsize=(9, 5.5))
        metric_df[["Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
            kind="bar",
            ax=ax,
            width=0.7,
            colormap="viridis",
        )
        ax.set_title("Day 4: Logistic Regression Evaluation (Precision, Recall, F1)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=0, fontsize=10, fontweight="semibold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        metrics_plot_path = os.path.join(self.reports_dir, "logistic_regression_metrics.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved metrics comparison chart to: {metrics_plot_path}")

        # 2. Confusion Matrices (Standard vs Balanced)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (name, y_pred) in zip(axes, self.predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"Confusion Matrix:\n{name}", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Retained (0)", "Churned (1)"])
            ax.set_yticklabels(["Retained (0)", "Churned (1)"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12,
                        fontweight="bold",
                    )
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
        cm_plot_path = os.path.join(self.reports_dir, "logistic_regression_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved confusion matrices to: {cm_plot_path}")

        # 3. ROC Curve
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for name, y_prob in self.probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess (0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax.set_title("Logistic Regression ROC Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        roc_plot_path = os.path.join(self.reports_dir, "logistic_regression_roc_curve.png")
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved ROC curve to: {roc_plot_path}")

    def save_artifacts(self, feature_names: list, scaler: object):
        """Saves model binary, scaler, and metadata."""
        best_model = self.models["Logistic Regression (Balanced)"]
        model_path = os.path.join(self.models_dir, "logistic_regression_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"[+] Saved Logistic Regression model binary to: {model_path}")

        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        metadata = {
            "author": "Rajarsh",
            "task": "Day 4: Train Logistic Regression and evaluate Precision, Recall, F1",
            "model_type": "Logistic Regression (Class-Balanced)",
            "metrics": self.metrics,
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        }
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Saved model metadata to: {meta_path}")


def main():
    # 1. Load Data
    df = load_raw_data()

    # 2. Feature Engineering & Preprocessing
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)

    # 3. Train & Evaluate
    pipeline = LogisticRegressionPipeline()
    results_df = pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
    pipeline.generate_plots(y_test)
    pipeline.save_artifacts(features, fe.scaler)

    print("\n" + "=" * 72)
    print("  DAY 4 TASK SUMMARY TABLE")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
