from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Reuse the exact model/data configuration already used by the project.
from src.model_comparison import (
    FEATURE_COLUMNS_PATH,
    INPUT_DIR,
    X_TEST_PATH,
    X_TRAIN_PATH,
    Y_TEST_PATH,
    Y_TRAIN_PATH,
    build_models,
    get_feature_names,
    load_features,
    load_target,
    require_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed" / "week3_day3_evaluation"
REPORT_PATH = ROOT / "reports" / "week3_day3_model_evaluation.md"


def evaluate_existing_project_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    models, scale_pos_weight = build_models(y_train)
    rows: list[dict[str, float | str]] = []
    reports: list[str] = []
    confusion: dict[str, list[list[int]]] = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "fit_seconds": float(fit_seconds),
        }
        rows.append(row)
        reports.append(
            f"{'=' * 78}\n{name}\n{'=' * 78}\n"
            + classification_report(y_test, pred, digits=4, zero_division=0)
        )
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        confusion[name] = cm.tolist()
        print(
            f"{name}: accuracy={row['accuracy']:.4f}, precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, roc_auc={row['roc_auc']:.4f}"
        )

    results = pd.DataFrame(rows)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    if results[metric_cols].isna().any().any():
        raise ValueError("One or more evaluation metrics are NaN.")
    if not ((results[metric_cols] >= 0) & (results[metric_cols] <= 1)).all().all():
        raise ValueError("Evaluation metrics must be between 0 and 1.")

    return results, reports, confusion, scale_pos_weight


def write_report(
    results: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scale_pos_weight: float,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Week 3 Day 3 — Model Evaluation & Comparison",
        "",
        "## Objective",
        "",
        "Evaluate the churn classification models using Accuracy, Precision, Recall, F1-score, and ROC-AUC and provide a direct model comparison.",
        "",
        "## Evaluation setup",
        "",
        f"- Training data: `{X_train.shape[0]:,} rows × {X_train.shape[1]:,} features`",
        f"- Held-out test data: `{X_test.shape[0]:,} rows × {X_test.shape[1]:,} features`",
        f"- Training churn rate: `{y_train.mean():.4f}`",
        f"- Test churn rate: `{y_test.mean():.4f}`",
        "- Same held-out test set used for every model.",
        "- Random state: `42`.",
        f"- XGBoost scale_pos_weight: `{scale_pos_weight:.4f}`.",
        "",
        "## Model evaluation results",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in results.iterrows():
        lines.append(
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The table reports the measured performance of all three models on the same held-out test set. No single metric is treated as a universal decision rule; the choice of model should depend on the business cost of false positives versus false negatives and the team's deployment requirements.",
        "",
        "## Generated artifacts",
        "",
        "- `data/processed/week3_day3_evaluation/model_evaluation_metrics.csv`",
        "- `data/processed/week3_day3_evaluation/classification_reports.txt`",
        "- `data/processed/week3_day3_evaluation/confusion_matrices.json`",
        "- `data/processed/week3_day3_evaluation/model_comparison.png`",
        "- `data/processed/week3_day3_evaluation/evaluation_summary.txt`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 78)
    print("CUSTOMER CHURN — WEEK 3 DAY 3 MODEL EVALUATION & COMPARISON")
    print("=" * 78)

    require_inputs()
    X_train = load_features(X_TRAIN_PATH)
    X_test = load_features(X_TEST_PATH)
    y_train = load_target(Y_TRAIN_PATH)
    y_test = load_target(Y_TEST_PATH)

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature-count mismatch.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature/target row counts do not match.")

    feature_names = get_feature_names()
    if len(feature_names) != X_train.shape[1]:
        raise ValueError("Feature-name count does not match X_train column count.")

    results, reports, confusion, scale_pos_weight = evaluate_existing_project_models(
        X_train, y_train, X_test, y_test
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "model_evaluation_metrics.csv", index=False)
    (OUTPUT_DIR / "classification_reports.txt").write_text("\n\n".join(reports), encoding="utf-8")
    (OUTPUT_DIR / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2), encoding="utf-8")

    plot_df = results.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    ax = plot_df.plot(kind="bar", figsize=(12, 7))
    ax.set_title("Customer Churn — Model Evaluation Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary_lines = [
        "WEEK 3 DAY 3 — MODEL EVALUATION SUMMARY",
        "",
        f"Test rows: {len(y_test):,}",
        f"Models evaluated: {len(results)}",
        "Metrics: Accuracy, Precision, Recall, F1, ROC-AUC",
        "",
        results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False),
        "",
        "All five requested metrics were generated successfully for every model.",
        "",
        "SUCCESS",
    ]
    (OUTPUT_DIR / "evaluation_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    write_report(results, X_train, X_test, y_train, y_test, scale_pos_weight)

    print("\nMODEL EVALUATION & COMPARISON")
    print("=" * 78)
    print(results[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Saved report to: {REPORT_PATH}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
