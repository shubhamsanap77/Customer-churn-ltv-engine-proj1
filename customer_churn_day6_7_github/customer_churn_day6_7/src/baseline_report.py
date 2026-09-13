import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import REPORTS_DIR, BASELINE_REPORT


def churn_rate(df, column, target="Churn"):
    table = (
        df.groupby(column)[target]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "churned_customers", "mean": "churn_rate"})
    )
    table["churn_rate"] = (table["churn_rate"] * 100).round(2)
    return table


def save_plot_churn_distribution(df):
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Churn")
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_distribution.png", dpi=150)
    plt.close()


def save_plot_by_category(df, column, filename, title):
    if column not in df.columns:
        return

    table = churn_rate(df, column)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=table, x=column, y="churn_rate")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=150)
    plt.close()


def add_tenure_group(df):
    df = df.copy()
    if "tenure" not in df.columns:
        return df

    bins = [-1, 6, 12, 24, 48, 72, float("inf")]
    labels = ["0-6", "7-12", "13-24", "25-48", "49-72", "73+"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def generate_report(result):
    df = result["clean"]
    missing = result["missing"]
    numeric_cols = result["numeric_cols"]
    categorical_cols = result["categorical_cols"]

    report_lines = []
    report_lines.append("# Baseline Analytics Report – Customer Churn")
    report_lines.append("")
    report_lines.append("## 1. Dataset Overview")
    report_lines.append("")
    report_lines.append(f"- Rows after duplicate removal: **{len(df):,}**")
    report_lines.append(f"- Columns after preprocessing: **{len(df.columns):,}**")
    report_lines.append(f"- Numeric features: **{len(numeric_cols)}**")
    report_lines.append(f"- Categorical features: **{len(categorical_cols)}**")
    report_lines.append("")

    churned = int(df["Churn"].sum())
    total = len(df)
    churn_rate_value = churned / total * 100 if total else 0

    report_lines.append("## 2. Baseline Churn KPI")
    report_lines.append("")
    report_lines.append(f"- Total customers: **{total:,}**")
    report_lines.append(f"- Churned customers: **{churned:,}**")
    report_lines.append(f"- Overall churn rate: **{churn_rate_value:.2f}%**")
    report_lines.append("")

    report_lines.append("## 3. Missing Values")
    report_lines.append("")
    missing.to_csv(REPORTS_DIR / "missing_values_before_after.csv")
    report_lines.append("| Column | Missing Before | Missing After |")
    report_lines.append("|---|---:|---:|")
    for idx, row in missing.iterrows():
        report_lines.append(
            f"| {idx} | {int(row['missing_before'])} | {int(row['missing_after'])} |"
        )
    report_lines.append("")

    report_lines.append("## 4. Numeric Summary")
    report_lines.append("")
    numeric_summary = df[numeric_cols].describe().T.round(2)
    numeric_summary.to_csv(REPORTS_DIR / "numeric_summary.csv")
    report_lines.append(numeric_summary.to_markdown())
    report_lines.append("")

    report_lines.append("## 5. Categorical Summary")
    report_lines.append("")
    categorical_summary = pd.DataFrame({
        "column": categorical_cols,
        "unique_values": [df[c].nunique() for c in categorical_cols],
        "missing_after": [df[c].isna().sum() for c in categorical_cols],
    })
    categorical_summary.to_csv(REPORTS_DIR / "categorical_summary.csv", index=False)
    report_lines.append(categorical_summary.to_markdown(index=False))
    report_lines.append("")

    # Important business segments
    df2 = add_tenure_group(df)

    segment_columns = [
        ("Contract", "churn_by_contract.csv"),
        ("InternetService", "churn_by_internet_service.csv"),
        ("PaymentMethod", "churn_by_payment_method.csv"),
        ("tenure_group", "churn_by_tenure_group.csv"),
        ("SeniorCitizen", "churn_by_senior_citizen.csv"),
    ]

    report_lines.append("## 6. Churn by Customer Segment")
    report_lines.append("")

    for column, filename in segment_columns:
        if column not in df2.columns:
            continue

        table = churn_rate(df2, column)
        table.to_csv(REPORTS_DIR / filename, index=False)

        report_lines.append(f"### {column}")
        report_lines.append("")
        report_lines.append(table.to_markdown(index=False))
        report_lines.append("")

    save_plot_churn_distribution(df2)
    save_plot_by_category(
        df2, "Contract", "churn_by_contract.png",
        "Churn Rate by Contract Type"
    )
    save_plot_by_category(
        df2, "InternetService", "churn_by_internet_service.png",
        "Churn Rate by Internet Service"
    )
    save_plot_by_category(
        df2, "PaymentMethod", "churn_by_payment_method.png",
        "Churn Rate by Payment Method"
    )
    save_plot_by_category(
        df2, "tenure_group", "churn_by_tenure_group.png",
        "Churn Rate by Tenure Group"
    )

    report_lines.append("## 7. Baseline Findings")
    report_lines.append("")
    report_lines.append(
        "The overall churn rate above is the baseline KPI for later predictive-model evaluation."
    )
    report_lines.append(
        "Segment tables identify customer groups with comparatively higher or lower churn."
    )
    report_lines.append(
        "The encoded dataset is ready for the Week 2 feature-engineering/modeling stage."
    )
    report_lines.append("")

    BASELINE_REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    return BASELINE_REPORT


if __name__ == "__main__":
    from .preprocessing import run_preprocessing
    result = run_preprocessing()
    path = generate_report(result)
    print(f"Report saved to: {path}")
