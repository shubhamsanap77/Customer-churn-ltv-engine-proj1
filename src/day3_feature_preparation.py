from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "day3_ml"

TARGETS = ["Churn", "churn", "CHURN", "Churn_Label", "churn_label",
           "Churn Value", "Churn_Value", "churn_value"]

def find_input():
    files = [p for p in PROCESSED_DIR.glob("*.csv")
             if p.stem.lower() not in {"x_train","x_test","y_train","y_test"}]
    if not files:
        raise FileNotFoundError(
            f"No engineered CSV found in {PROCESSED_DIR}. "
            "Place the Day 3 engineered dataset there."
        )
    preferred = [p for p in files if any(x in p.stem.lower()
                  for x in ("engineered","feature","dataset"))]
    return preferred[0] if preferred else files[0]

def target_column(df):
    normalized = {str(c).strip().lower().replace(" ","_"): c for c in df.columns}
    for t in TARGETS:
        key = t.lower().replace(" ","_")
        if key in normalized:
            return normalized[key]
    raise ValueError("Churn target not found. Columns: " + ", ".join(map(str,df.columns)))

def binary_target(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        vals = set(x.dropna().unique())
        if vals.issubset({0,1}):
            return x.astype("Int64")
        if len(vals) == 2:
            a,b = sorted(vals)
            return x.map({a:0,b:1}).astype("Int64")
    x = s.astype(str).str.strip().str.lower()
    out = x.map({"yes":1,"no":0,"true":1,"false":0,"1":1,"0":0,
                 "churned":1,"not churned":0})
    if out.isna().any():
        raise ValueError("Unknown churn values: " + str(sorted(x[out.isna()].unique())))
    return out.astype(int)

def main():
    print("="*60)
    print("CUSTOMER CHURN — DAY 3 FEATURE PREPARATION")
    print("="*60)

    path = find_input()
    df = pd.read_csv(path)
    target = target_column(df)
    y = binary_target(df[target])

    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    X = df.drop(columns=[target]).copy()

    ids = [c for c in X.columns if str(c).strip().lower().replace(" ","_")
           in {"id","customerid","customer_id"}]
    if ids:
        X = X.drop(columns=ids)

    for c in X.columns:
        if X[c].dtype == "object":
            converted = pd.to_numeric(X[c], errors="coerce")
            if X[c].notna().sum() and converted.notna().sum()/X[c].notna().sum() >= .95:
                X[c] = converted

    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    transformer = ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical)
    ])

    Xtr = pd.DataFrame(transformer.fit_transform(Xtr),
                       columns=transformer.get_feature_names_out())
    Xte = pd.DataFrame(transformer.transform(Xte),
                       columns=transformer.get_feature_names_out())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtr.to_csv(OUTPUT_DIR/"X_train.csv", index=False)
    Xte.to_csv(OUTPUT_DIR/"X_test.csv", index=False)
    ytr.to_csv(OUTPUT_DIR/"y_train.csv", index=False)
    yte.to_csv(OUTPUT_DIR/"y_test.csv", index=False)
    pd.DataFrame({"feature": transformer.get_feature_names_out()}).to_csv(
        OUTPUT_DIR/"feature_columns.csv", index=False)

    (OUTPUT_DIR/"dataset_summary.txt").write_text(
        f"Input: {path.name}\nTarget: {target}\n"
        f"Rows: {len(df)}\nFeatures before encoding: {X.shape[1]}\n"
        f"Features after encoding: {Xtr.shape[1]}\n"
        f"X_train: {Xtr.shape}\nX_test: {Xte.shape}\n"
        f"y_train: {ytr.shape}\ny_test: {yte.shape}\n"
        f"Test size: 20%\nRandom state: 42\n"
        f"Overall churn rate: {y.mean():.4f}\n"
        f"Train churn rate: {ytr.mean():.4f}\n"
        f"Test churn rate: {yte.mean():.4f}\n"
        f"Removed identifiers: {ids if ids else 'None'}\n",
        encoding="utf-8"
    )

    print(f"Input: {path.name}")
    print(f"Target: {target}")
    print(f"X_train: {Xtr.shape}")
    print(f"X_test : {Xte.shape}")
    print(f"y_train: {ytr.shape}")
    print(f"y_test : {yte.shape}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("SUCCESS")

if __name__ == "__main__":
    main()
