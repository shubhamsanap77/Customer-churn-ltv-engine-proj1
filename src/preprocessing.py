from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RAW_DIR, CLEANED_FILE, ENCODED_FILE


def find_input_csv() -> Path:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DIR}. "
            "Place the Telco Customer Churn CSV in data/raw/."
        )
    return csv_files[0]


def load_data() -> pd.DataFrame:
    path = find_input_csv()
    print(f"Loading dataset: {path.name}")
    return pd.read_csv(path)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    # Common Telco dataset fields
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # After standardization the name normally becomes TotalCharges.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    # Remove exact duplicates while keeping a record of the cleaned dataset.
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def get_target_column(df: pd.DataFrame) -> str:
    candidates = ["Churn", "churn", "CHURN"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError("Could not find the Churn target column in the dataset.")


def make_binary_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    values = df[target_col].astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
    converted = values.map(mapping)

    if converted.isna().any():
        unknown = sorted(values[converted.isna()].unique())
        raise ValueError(f"Unknown values in {target_col}: {unknown}")

    df[target_col] = converted.astype(int)
    return df


def preprocess(df: pd.DataFrame):
    target_col = get_target_column(df)
    df = make_binary_target(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Customer ID is an identifier, not a useful analytical feature.
    id_columns = [c for c in X.columns if c.lower() in {"customerid", "customer_id"}]
    X_model = X.drop(columns=id_columns, errors="ignore")

    numeric_cols = X_model.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_model.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    encoded_array = transformer.fit_transform(X_model)
    feature_names = transformer.get_feature_names_out()

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )

    encoded_df[target_col] = y.values

    # Clean human-readable dataset: impute without one-hot encoding.
    clean_df = X_model.copy()

    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())

    for col in categorical_cols:
        mode = clean_df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        clean_df[col] = clean_df[col].fillna(fill_value)

    clean_df[target_col] = y.values

    return clean_df, encoded_df, transformer, numeric_cols, categorical_cols


def run_preprocessing():
    df_raw = load_data()
    df_prepared = prepare_raw_data(df_raw)

    missing_before = (
        df_prepared.isna().sum()
        .rename("missing_before")
        .to_frame()
    )

    clean_df, encoded_df, transformer, numeric_cols, categorical_cols = preprocess(
        df_prepared
    )

    missing_after = (
        clean_df.isna().sum()
        .rename("missing_after")
        .to_frame()
    )

    missing_comparison = missing_before.join(missing_after, how="outer").fillna(0)
    missing_comparison["missing_before"] = missing_comparison["missing_before"].astype(int)
    missing_comparison["missing_after"] = missing_comparison["missing_after"].astype(int)

    clean_df.to_csv(CLEANED_FILE, index=False)
    encoded_df.to_csv(ENCODED_FILE, index=False)

    return {
        "raw": df_raw,
        "prepared": df_prepared,
        "clean": clean_df,
        "encoded": encoded_df,
        "missing": missing_comparison,
        "transformer": transformer,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print(f"Cleaned data saved to: {CLEANED_FILE}")
    print(f"Encoded data saved to: {ENCODED_FILE}")
