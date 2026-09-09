"""
Week 2: SHAP Model Explainability Module
Uses TreeExplainer to interpret model predictions globally and locally.
Generates beeswarm plots, bar importance charts, and individual customer waterfalls.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from week_2.data_loader import load_raw_data
    from week_2.feature_engineering import FeatureEngineer
    from week_2.train_and_evaluate import ModelTrainer
except ImportError:
    from data_loader import load_raw_data
    from feature_engineering import FeatureEngineer
    from train_and_evaluate import ModelTrainer


class ShapExplainer:
    """
    Computes and visualizes SHAP (SHapley Additive exPlanations) values for Churn models.
    """

    def __init__(self, model, feature_names: list, reports_dir: str = None):
        self.model = model
        self.feature_names = feature_names
        base_dir = os.path.dirname(__file__)
        self.reports_dir = reports_dir or os.path.join(base_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.explainer = None
        self.shap_values = None

    def explain(self, X_test: pd.DataFrame, sample_size: int = 500):
        """
        Computes SHAP values using TreeExplainer.
        """
        print("\n" + "=" * 70)
        print("  WEEK 2: COMPUTING SHAP EXPLAINABILITY VALUES")
        print("=" * 70)

        # Sample for fast, accurate SHAP computation
        if len(X_test) > sample_size:
            X_sample = X_test.sample(n=sample_size, random_state=42)
        else:
            X_sample = X_test

        print(f"[*] Explaining model with TreeExplainer on {len(X_sample)} test samples...")
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer(X_sample)

        # Handle binary classification output format
        if len(self.shap_values.shape) == 3 and self.shap_values.shape[2] == 2:
            # Multi-output: select positive churn class (index 1)
            explanation = self.shap_values[:, :, 1]
        else:
            explanation = self.shap_values

        return explanation, X_sample

    def generate_plots(self, explanation, X_sample):
        """Generates and saves global and local SHAP visual explanation artifacts."""
        print("[*] Generating SHAP summary and importance plots...")

        # 1. SHAP Beeswarm Summary Plot
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        plt.title("SHAP Feature Impact on Customer Churn (Beeswarm)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = os.path.join(self.reports_dir, "shap_summary_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Beeswarm plot to: {beeswarm_path}")

        # 2. SHAP Bar Feature Importance Plot
        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation, max_display=15, show=False)
        plt.title("SHAP Global Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        bar_path = os.path.join(self.reports_dir, "shap_feature_importance_bar.png")
        plt.savefig(bar_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Importance Bar plot to: {bar_path}")

        # 3. Local Explanations (Waterfall)
        # Find a high-churn-risk customer and a low-churn-risk customer
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_sample)[:, 1]
        else:
            probs = self.model.predict(X_sample)

        high_risk_idx = int(np.argmax(probs))
        low_risk_idx = int(np.argmin(probs))

        # High Risk Waterfall
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation[high_risk_idx], max_display=10, show=False)
        plt.title(f"Customer #{high_risk_idx} [HIGH RISK - Churn Prob: {probs[high_risk_idx]:.1%}]", fontsize=12, fontweight="bold")
        plt.tight_layout()
        high_risk_path = os.path.join(self.reports_dir, "shap_waterfall_high_risk.png")
        plt.savefig(high_risk_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved High-Risk Customer waterfall to: {high_risk_path}")

        # Low Risk Waterfall
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation[low_risk_idx], max_display=10, show=False)
        plt.title(f"Customer #{low_risk_idx} [LOW RISK - Churn Prob: {probs[low_risk_idx]:.1%}]", fontsize=12, fontweight="bold")
        plt.tight_layout()
        low_risk_path = os.path.join(self.reports_dir, "shap_waterfall_low_risk.png")
        plt.savefig(low_risk_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Low-Risk Customer waterfall to: {low_risk_path}")

        # 4. Save Top Drivers to JSON
        vals = explanation.values
        mean_abs_shap = np.abs(vals).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:10]
        top_drivers = [
            {"feature": X_sample.columns[i], "mean_abs_shap": round(float(mean_abs_shap[i]), 5)}
            for i in top_indices
        ]
        drivers_path = os.path.join(self.reports_dir, "top_shap_churn_drivers.json")
        with open(drivers_path, "w") as f:
            json.dump(top_drivers, f, indent=4)
        print(f"[+] Saved top SHAP churn drivers to: {drivers_path}")

        return top_drivers


if __name__ == "__main__":
    df = load_raw_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)

    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Use best tree model (or XGBoost specifically for optimal SHAP tree metrics)
    target_model = trainer.models.get("XGBoost", trainer.best_model)
    explainer = ShapExplainer(target_model, features)
    explanation, X_sample = explainer.explain(X_test)
    drivers = explainer.generate_plots(explanation, X_sample)

    print("\n[+] Top 10 Churn Drivers by Mean |SHAP| Impact:")
    for d in drivers:
        print(f"    - {d['feature']:<30}: {d['mean_abs_shap']:.5f}")
