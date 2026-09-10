"""
Week 2: Classification Threshold & Business Profit Optimizer
Analyzes the decision threshold spectrum [0.05 to 0.95].
Finds optimal operating thresholds for maximum F1-Score and maximum Net Profit.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

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


class ThresholdOptimizer:
    """
    Optimizes classification thresholds for statistical balance and business profit.
    """

    def __init__(
        self,
        retention_cost: float = 50.0,
        customer_value_saved: float = 540.0,
        retention_success_rate: float = 0.50,
        reports_dir: str = None,
    ):
        self.retention_cost = retention_cost
        self.customer_value_saved = customer_value_saved
        self.retention_success_rate = retention_success_rate

        base_dir = os.path.dirname(__file__)
        self.reports_dir = reports_dir or os.path.join(base_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    def compute_profit(self, tp: int, fp: int, fn: int, tn: int) -> float:
        """
        Calculates net business value based on retention campaign economics:
        - TP: Reach out to actual churner; success_rate% stay, minus outreach cost.
        - FP: Reach out to non-churner; pay outreach cost unnecessarily.
        - FN: Missed churner; lose full customer value.
        """
        net_tp_gain = (self.retention_success_rate * self.customer_value_saved) - self.retention_cost
        fp_loss = self.retention_cost
        fn_loss = self.customer_value_saved

        return (tp * net_tp_gain) - (fp * fp_loss) - (fn * fn_loss)

    def optimize(self, y_true: pd.Series, y_probs: np.ndarray) -> dict:
        """
        Scans thresholds from 0.05 to 0.95 and evaluates metric tradeoffs and business net profit.
        """
        print("\n" + "=" * 70)
        print("  WEEK 2: THRESHOLD & PROFIT OPTIMIZATION ANALYSIS")
        print("=" * 70)

        thresholds = np.linspace(0.05, 0.95, 91)
        precisions, recalls, f1s, profits = [], [], [], []

        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            profit = self.compute_profit(tp, fp, fn, tn)

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            profits.append(profit)

        # Optimal points
        opt_f1_idx = int(np.argmax(f1s))
        opt_profit_idx = int(np.argmax(profits))
        default_idx = int(np.argmin(np.abs(thresholds - 0.50)))

        opt_f1_thresh = float(thresholds[opt_f1_idx])
        opt_profit_thresh = float(thresholds[opt_profit_idx])

        print(f"[*] Default Threshold (0.50):")
        print(f"    - F1-Score:    {f1s[default_idx]:.4f}")
        print(f"    - Recall:      {recalls[default_idx]:.4f}")
        print(f"    - Precision:   {precisions[default_idx]:.4f}")
        print(f"    - Net Value:   ${profits[default_idx]:,.2f}")

        print(f"\n[+] Optimal F1 Threshold ({opt_f1_thresh:.2f}):")
        print(f"    - F1-Score:    {f1s[opt_f1_idx]:.4f} (+{(f1s[opt_f1_idx]-f1s[default_idx])*100:.2f} pts)")
        print(f"    - Recall:      {recalls[opt_f1_idx]:.4f}")
        print(f"    - Precision:   {precisions[opt_f1_idx]:.4f}")

        print(f"\n[+] Optimal Business Profit Threshold ({opt_profit_thresh:.2f}):")
        print(f"    - Net Value:   ${profits[opt_profit_idx]:,.2f} (+${profits[opt_profit_idx]-profits[default_idx]:,.2f} gain)")
        print(f"    - F1-Score:    {f1s[opt_profit_idx]:.4f}")
        print(f"    - Recall:      {recalls[opt_profit_idx]:.4f}")

        results = {
            "default_threshold_0.50": {
                "threshold": 0.50,
                "precision": round(float(precisions[default_idx]), 4),
                "recall": round(float(recalls[default_idx]), 4),
                "f1_score": round(float(f1s[default_idx]), 4),
                "net_profit": round(float(profits[default_idx]), 2),
            },
            "optimal_f1_threshold": {
                "threshold": round(opt_f1_thresh, 2),
                "precision": round(float(precisions[opt_f1_idx]), 4),
                "recall": round(float(recalls[opt_f1_idx]), 4),
                "f1_score": round(float(f1s[opt_f1_idx]), 4),
                "net_profit": round(float(profits[opt_f1_idx]), 2),
            },
            "optimal_profit_threshold": {
                "threshold": round(opt_profit_thresh, 2),
                "precision": round(float(precisions[opt_profit_idx]), 4),
                "recall": round(float(recalls[opt_profit_idx]), 4),
                "f1_score": round(float(f1s[opt_profit_idx]), 4),
                "net_profit": round(float(profits[opt_profit_idx]), 2),
            },
        }

        # Save configuration
        config_path = os.path.join(self.reports_dir, "optimal_threshold_config.json")
        with open(config_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[+] Saved threshold config to: {config_path}")

        # Plots
        self._plot_metrics_curve(thresholds, precisions, recalls, f1s, opt_f1_thresh)
        self._plot_profit_curve(thresholds, profits, opt_profit_thresh)

        return results

    def _plot_metrics_curve(self, thresholds, precisions, recalls, f1s, opt_f1):
        """Plots Precision, Recall, and F1 against threshold."""
        plt.figure(figsize=(9, 5.5))
        plt.plot(thresholds, precisions, label="Precision", color="#2a9d8f", lw=2)
        plt.plot(thresholds, recalls, label="Recall", color="#e76f51", lw=2)
        plt.plot(thresholds, f1s, label="F1-Score", color="#264653", lw=2.5)
        plt.axvline(x=0.50, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)")
        plt.axvline(x=opt_f1, color="#e63946", linestyle="-.", lw=1.8, label=f"Optimal F1 ({opt_f1:.2f})")

        plt.title("Precision, Recall & F1-Score vs. Decision Threshold", fontsize=13, fontweight="bold")
        plt.xlabel("Classification Threshold", fontsize=11)
        plt.ylabel("Score", fontsize=11)
        plt.ylim(0, 1.05)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, "precision_recall_threshold_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved threshold curve to: {plot_path}")

    def _plot_profit_curve(self, thresholds, profits, opt_profit):
        """Plots net business profit vs threshold."""
        plt.figure(figsize=(9, 5.5))
        plt.plot(thresholds, profits, color="#1d3557", lw=2.5, label="Net Business Value ($)")
        plt.axvline(x=0.50, color="gray", linestyle="--", alpha=0.7, label="Default (0.50)")
        plt.axvline(x=opt_profit, color="#2a9d8f", linestyle="-.", lw=1.8, label=f"Max Profit ({opt_profit:.2f})")

        plt.title("Business Value / Retention ROI vs Decision Threshold", fontsize=13, fontweight="bold")
        plt.xlabel("Classification Threshold", fontsize=11)
        plt.ylabel("Expected Net Value ($)", fontsize=11)
        plt.legend(loc="lower center")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.reports_dir, "profit_curve_by_threshold.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[+] Saved profit curve to: {plot_path}")


if __name__ == "__main__":
    df = load_raw_data()
    fe = FeatureEngineer()
    X_train, X_test, y_train, y_test, features = fe.fit_transform(df)

    trainer = ModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    y_probs = trainer.probabilities[trainer.best_model_name]

    optimizer = ThresholdOptimizer()
    optimizer.optimize(y_test, y_probs)
