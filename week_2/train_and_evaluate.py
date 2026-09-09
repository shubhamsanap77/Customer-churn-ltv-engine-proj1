"""
Week 2: Model Training & Evaluation Pipeline
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
Evaluates models using Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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


class ModelTrainer:
    """
    Manages model training, evaluation, comparison, artifact persistence, and visualization.
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
        self.best_model_name = None
        self.best_model = None

    def initialize_models(self, pos_weight: float = 2.77):
        """Initializes standard baseline, ensemble, and boosting classifiers."""
        self.models = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced",
                C=0.1,
                max_iter=1000,
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                scale_pos_weight=pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains all models and evaluates Precision, Recall, F1, Accuracy, and ROC-AUC.
        """
        # Calculate positive class weighting (Neg / Pos)
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        pos_weight = float(neg_count / max(pos_count, 1))

        self.initialize_models(pos_weight=pos_weight)

        print("\n" + "=" * 70)
        print("  WEEK 2: MODEL TRAINING & EVALUATION BENCHMARK")
        print("=" * 70)

        best_score = -1.0

        for name, model in self.models.items():
            print(f"\n[*] Training {name}...")
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            self.predictions[name] = y_pred
            self.probabilities[name] = y_prob

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

            print(f"    - Accuracy:  {acc:.4f}")
            print(f"    - Precision: {prec:.4f}")
            print(f"    - Recall:    {rec:.4f}")
            print(f"    - F1-Score:  {f1:.4f}")
            print(f"    - ROC-AUC:   {auc:.4f}")

            # Optimize for high F1 and Recall (critical for churn retention)
            composite_score = 0.6 * f1 + 0.4 * auc
            if composite_score > best_score:
                best_score = composite_score
                self.best_model_name = name
                self.best_model = model

        print(f"\n[+] Champion Model Selected: {self.best_model_name} (Composite F1/AUC: {best_score:.4f})")
        return self.metrics

    def generate_comparison_table(self) -> pd.DataFrame:
        """Returns comparison DataFrame sorted by F1-Score."""
        df = pd.DataFrame(self.metrics).T
        df = df.sort_values(by="F1-Score", ascending=False)
        return df

    def save_plots(self, y_test: pd.Series):
        """Generates and saves visual evaluation reports."""
        plt.style.use("tableau-colorblind10" if "tableau-colorblind10" in plt.style.available else "default")

        # 1. Metric Comparison Bar Chart
        metric_df = pd.DataFrame(self.metrics).T
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_title("Week 2 Model Comparison (Precision, Recall, F1, ROC-AUC, Accuracy)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=0, fontsize=10, fontweight="semibold")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        metrics_plot_path = os.path.join(self.reports_dir, "model_metrics_comparison.png")
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved metrics chart to: {metrics_plot_path}")

        # 2. Confusion Matrices
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (name, y_pred) in zip(axes, self.predictions.items()):
            cm = confusion_matrix(y_test, y_pred)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"CM: {name}", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Retained (0)", "Churned (1)"])
            ax.set_yticklabels(["Retained (0)", "Churned (1)"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

            # Annotate numbers in confusion matrix
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
        cm_plot_path = os.path.join(self.reports_dir, "confusion_matrices_comparison.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved confusion matrices to: {cm_plot_path}")

        # 3. Combined ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, y_prob in self.probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess (0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
        ax.set_title("Week 2 ROC Curves Comparison", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        roc_plot_path = os.path.join(self.reports_dir, "roc_curves_comparison.png")
        plt.savefig(roc_plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved ROC curves to: {roc_plot_path}")

    def save_artifacts(self, feature_names: list, scaler: object):
        """Saves model binaries, preprocessors, and metadata."""
        # 1. Save champion model
        model_pkl_path = os.path.join(self.models_dir, "best_churn_model.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(self.best_model, f)
        print(f"[+] Saved champion model binary to: {model_pkl_path}")

        # Also copy to root models/ directory
        root_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(root_models_dir, exist_ok=True)
        root_model_pkl = os.path.join(root_models_dir, "best_churn_model.pkl")
        with open(root_model_pkl, "wb") as f:
            pickle.dump(self.best_model, f)

        # 2. Save Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # 3. Save Metadata
        metadata = {
            "champion_model": self.best_model_name,
            "metrics": self.metrics,
            "feature_names": feature_names,
            "total_features": len(feature_names),
        }
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[+] Saved model metadata to: {meta_path}")

        # 4. Save CSV report
        report_df = self.generate_comparison_table()
        csv_path = os.path.join(self.reports_dir, "model_benchmark_results.csv")
        report_df.to_csv(csv_path)
        print(f"[+] Saved benchmark CSV to: {csv_path}")


if __name__ == "__main__":
    df = load_raw_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)

    trainer = ModelTrainer()
    metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    trainer.save_plots(y_test)
    trainer.save_artifacts(features, fe.scaler)

    print("\n[+] Benchmark Summary Table:")
    print(trainer.generate_comparison_table())
