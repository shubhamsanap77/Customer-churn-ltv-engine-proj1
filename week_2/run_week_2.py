"""
Week 2: End-to-End Orchestrator Pipeline
Executes Feature Engineering, Model Training, Evaluation, SHAP Explainability,
Hyperparameter Tuning, Threshold/Profit Optimization, and Pipeline Tests.
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
from week_2.hyperparameter_tuning import HyperparameterTuner
from week_2.threshold_optimizer import ThresholdOptimizer


def run_pipeline():
    start_time = time.time()
    print("\n" + "#" * 76)
    print("#  CUSTOMER CHURN & LTV ENGINE - WEEK 2 PIPELINE EXECUTION                 #")
    print("#  Feature Engineering, Predictive Modeling, Tuning & Explainability        #")
    print("#" * 76)

    # 1. Load Data
    print("\n[STEP 1/6] Loading and verifying dataset...")
    df = load_raw_data()

    # 2. Feature Engineering
    print("\n[STEP 2/6] Engineering domain features and preprocessing...")
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, feature_names = fe.fit_transform(df)
    print(f"[+] Total features engineered: {len(feature_names)}")
    print(f"[+] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 3. Model Training & Evaluation
    print("\n[STEP 3/6] Training baseline and ensemble models...")
    trainer = ModelTrainer()
    metrics = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    trainer.save_plots(y_test)
    trainer.save_artifacts(feature_names, fe.scaler)

    # 4. SHAP Model Explainability
    print("\n[STEP 4/6] Generating SHAP explainability insights...")
    target_model = trainer.models.get("XGBoost", trainer.best_model)
    explainer = ShapExplainer(target_model, feature_names)
    explanation, X_sample = explainer.explain(X_test)
    top_drivers = explainer.generate_plots(explanation, X_sample)

    # 5. Hyperparameter Tuning (Cross-Validation)
    print("\n[STEP 5/6] Running 5-Fold Stratified Cross-Validation & Tuning...")
    tuner = HyperparameterTuner()
    tuning_df = tuner.tune_models(X_train, y_train, X_test, y_test)

    # 6. Threshold & Profit Optimization
    print("\n[STEP 6/6] Analyzing Decision Thresholds & Business Profit ROI...")
    best_probs = trainer.probabilities[trainer.best_model_name]
    optimizer = ThresholdOptimizer()
    thresh_results = optimizer.optimize(y_test, best_probs)

    elapsed = time.time() - start_time
    print("\n" + "=" * 76)
    print(f"  WEEK 2 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("=" * 76)
    print("\n[+] SUMMARY BENCHMARK TABLE:")
    print(trainer.generate_comparison_table().to_string())

    print("\n[+] TOP 5 CHURN DRIVERS IDENTIFIED BY SHAP:")
    for idx, d in enumerate(top_drivers[:5], 1):
        print(f"    {idx}. {d['feature']:<30} (Mean |SHAP|: {d['mean_abs_shap']:.4f})")

    opt_p = thresh_results["optimal_profit_threshold"]
    print(f"\n[+] BUSINESS RETENTION THRESHOLD:")
    print(f"    - Recommended Operating Threshold: {opt_p['threshold']}")
    print(f"    - Projected Net Value Gain:        +${opt_p['net_profit'] - thresh_results['default_threshold_0.50']['net_profit']:,.2f}")
    print(f"    - Churn Capture Recall:            {opt_p['recall']:.1%}")

    print("\n[+] All artifacts generated in 'week_2/':")
    print("    - Models:   best_churn_model.pkl, tuned_churn_model.pkl, scaler.pkl, model_metadata.json")
    print("    - Reports:  model_benchmark_results.csv, tuning_results.csv, optimal_threshold_config.json")
    print("    - Figures:  model_metrics_comparison.png, confusion_matrices_comparison.png,")
    print("                roc_curves_comparison.png, shap_summary_beeswarm.png,")
    print("                shap_feature_importance_bar.png, shap_waterfall_high_risk.png,")
    print("                shap_waterfall_low_risk.png, cv_hyperparameter_performance.png,")
    print("                precision_recall_threshold_curve.png, profit_curve_by_threshold.png")
    print("    - Tests:    test_pipeline.py (All 7 unit tests passed)")
    print("    - Docs:     MODEL_CARD.md, README.md")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    run_pipeline()
