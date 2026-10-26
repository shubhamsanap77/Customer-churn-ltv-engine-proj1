from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "processed" / "day3_ml"
OUTPUT_DIR = ROOT / "data" / "processed" / "day6_model_comparison"
REPORT_PATH = ROOT / "reports" / "day6_model_comparison.md"

X_TRAIN_PATH = INPUT_DIR / "X_train.csv"
X_TEST_PATH = INPUT_DIR / "X_test.csv"
Y_TRAIN_PATH = INPUT_DIR / "y_train.csv"
Y_TEST_PATH = INPUT_DIR / "y_test.csv"
FEATURE_COLUMNS_PATH = INPUT_DIR / "feature_columns.csv"


def require_inputs() -> None:
    required = [X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH, Y_TEST_PATH, FEATURE_COLUMNS_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Day 6 input files are missing. Run Day 3 first and make sure the generated files exist under data/processed/day3_ml/.\n"
            + "\n".join(f"- {p}" for p in missing)
        )


def load_target(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if df.shape[1] != 1:
        raise ValueError(f"Expected one target column in {path}, found {df.shape[1]} columns.")
    y = pd.to_numeric(df.iloc[:, 0], errors="raise").astype(np.int8)
    vals = sorted(y.dropna().unique().tolist())
    if not set(vals).issubset({0, 1}):
        raise ValueError(f"Target in {path} must contain only 0/1. Found: {vals}")
    return y


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=np.float32)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if df.shape[1] == 0:
        raise ValueError(f"No feature columns found in {path}")
    return df


def get_feature_names() -> list[str]:
    df = pd.read_csv(FEATURE_COLUMNS_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df["feature"].astype(str).tolist() if "feature" in df.columns else df.iloc[:, 0].astype(str).tolist()


def build_models(y_train: pd.Series) -> tuple[dict[str, object], float]:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("Training target contains no positive churn cases.")

    scale_pos_weight = negatives / positives
    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear", max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models, scale_pos_weight


def evaluate_model(name: str, model: object, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, str]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "train_seconds": train_seconds,
    }
    report = classification_report(y_test, predictions, digits=4, zero_division=0)
    return metrics, report


def write_report(results_df: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, scale_pos_weight: float) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = results_df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    table["train_seconds"] = table["train_seconds"].map(lambda x: f"{x:.2f}")

    lines = [
        "# Day 6 — XGBoost Training & Model Comparison",
        "",
        "## Scope",
        "",
        "Week 2 Day 4–6 requires Logistic Regression, Random Forest, and XGBoost, evaluated with precision, recall, and F1-score.",
        "",
        "## Data used",
        "",
        f"- Training features: `{X_train.shape[0]:,} rows × {X_train.shape[1]:,} features`",
        f"- Test features: `{X_test.shape[0]:,} rows × {X_test.shape[1]:,} features`",
        f"- Training churn rate: `{y_train.mean():.4f}`",
        f"- Test churn rate: `{y_test.mean():.4f}`",
        "- Random state: `42`",
        f"- XGBoost `scale_pos_weight`: `{scale_pos_weight:.4f}`",
        "",
        "## Model comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['accuracy']} | {row['precision']} | {row['recall']} | {row['f1']} | {row['roc_auc']} | {row['train_seconds']} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `data/processed/day6_model_comparison/model_metrics.csv`",
        "- `data/processed/day6_model_comparison/classification_reports.txt`",
        "- `data/processed/day6_model_comparison/xgboost_feature_importance.csv`",
        "- `data/processed/day6_model_comparison/model_comparison.png`",
        "- `data/processed/day6_model_comparison/model_config.json`",
        "",
        "## Notes",
        "",
        "All three models use the same Day 3 train/test split. The requested precision, recall, and F1 metrics are reported on the held-out test set; accuracy and ROC-AUC are included as additional diagnostics.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("CUSTOMER CHURN — DAY 6 XGBOOST & MODEL COMPARISON")
    print("=" * 72)

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

    models, scale_pos_weight = build_models(y_train)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    classification_blocks = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, report = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        all_metrics.append(metrics)
        classification_blocks.append(f"\n{'=' * 72}\n{name}\n{'=' * 72}\n{report}")
        print(
            f"{name}: precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(all_metrics).sort_values("f1", ascending=False).reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    (OUTPUT_DIR / "classification_reports.txt").write_text("".join(classification_blocks), encoding="utf-8")

    xgb_importance = models["XGBoost"].feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": xgb_importance})
    importance_df.sort_values("importance", ascending=False).head(50).to_csv(
        OUTPUT_DIR / "xgboost_feature_importance.csv", index=False
    )

    plot_df = results_df.set_index("model")[["precision", "recall", "f1"]]
    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Customer Churn — Classification Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    config = {
        "random_state": 42,
        "xgboost_scale_pos_weight": scale_pos_weight,
        "xgboost_parameters": {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.5,
            "reg_lambda": 1.0,
            "tree_method": "hist",
        },
    }
    (OUTPUT_DIR / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report(results_df, X_train, X_test, y_train, y_test, scale_pos_weight)

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(results_df[["model", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")
    print(f"Saved report to: {REPORT_PATH}")
    print("SUCCESS")


if __name__ == "__main__":
    main()
