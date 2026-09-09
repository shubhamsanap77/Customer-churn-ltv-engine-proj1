"""
Week 2: End-to-End Orchestrator Pipeline
Executes the full Feature Engineering, Model Training, Evaluation, and SHAP Explainability tasks.
"""

import os
import sys
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from week_2.data_loader import load_raw_data
from week_2.feature_engineering import FeatureEngineer
from week_2.train_and_evaluate import ModelTrainer
from week_2.shap_explainability import ShapExplainer


def run_pipeline():
    start_time = time.time()
    print("\n" + "#" * 76)
    print("#  CUSTOMER CHURN & LTV ENGINE - WEEK 2 PIPELINE EXECUTION                 #")
    print("#  Feature Engineering, Predictive Modeling & SHAP Explainability          #")
    print("#" * 76)

    # 1. Load Data
    print("\n[STEP 1/4] Loading and verifying dataset...")
    df = load_raw_data()

    # 2. Feature Engineering
    print("\n[STEP 2/4] Engineering domain features and preprocessing...")
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, feature_names = fe.fit_transform(df)
    print(f"[+] Total features engineered: {len(feature_names)}")
    print(f"[+] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Model Training & Evaluation
    print("\n[STEP 3/4] Training models & benchmarking metrics...")
    trainer = ModelTrainer()
    metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    trainer.save_plots(y_test)
    trainer.save_artifacts(feature_names, fe.scaler)

    # 4. SHAP Model Explainability
    print("\n[STEP 4/4] Generating SHAP explainability insights...")
    target_model = trainer.models.get("XGBoost", trainer.best_model)
    explainer = ShapExplainer(target_model, feature_names)
    explanation, X_sample = explainer.explain(X_test)
    top_drivers = explainer.generate_plots(explanation, X_sample)

    elapsed = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"  WEEK 2 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 76)
    print("\n[+] SUMMARY BENCHMARK TABLE:")
    print(trainer.generate_comparison_table().to_string())
    print("\n[+] TOP 5 CHURN DRIVERS IDENTIFIED BY SHAP:")
    for idx, d in enumerate(top_drivers[:5], 1):
        print(f"    {idx}. {d['feature']:<30} (Mean |SHAP|: {d['mean_abs_shap']:.4f})")

    print("\n[+] Generated Deliverables in 'week_2/':")
    print("    - Models:   week_2/models/best_churn_model.pkl, scaler.pkl, model_metadata.json")
    print("    - Reports:  week_2/reports/model_benchmark_results.csv")
    print("    - Figures:  week_2/reports/model_metrics_comparison.png")
    print("                week_2/reports/confusion_matrices_comparison.png")
    print("                week_2/reports/roc_curves_comparison.png")
    print("                week_2/reports/shap_summary_beeswarm.png")
    print("                week_2/reports/shap_feature_importance_bar.png")
    print("                week_2/reports/shap_waterfall_high_risk.png")
    print("                week_2/reports/shap_waterfall_low_risk.png")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_pipeline()
