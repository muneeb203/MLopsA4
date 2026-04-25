import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def preprocessing(input_path: str, output_dir: str) -> dict:
    """
    Advanced preprocessing:
    - Drop extremely sparse columns (>90% missing)
    - Numeric imputation: median
    - Categorical imputation: mode then label encode
    - High-cardinality columns: frequency encoding
    """
    print("=" * 50)
    print("STEP 3: PREPROCESSING")
    print("=" * 50)

    df = pd.read_csv(input_path)
    print(f"Input shape: {df.shape}")

    # Drop columns with >90% missing
    thresh = 0.9 * len(df)
    cols_before = df.shape[1]
    df = df.dropna(thresh=len(df) - thresh, axis=1)
    print(f"Dropped {cols_before - df.shape[1]} columns (>90% missing)")

    # Separate features and target
    target = df["isFraud"].copy()
    df = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric cols: {len(numeric_cols)}, Categorical cols: {len(cat_cols)}")

    # Numeric imputation: median strategy
    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Categorical handling
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")
        n_unique = df[col].nunique()

        if n_unique > 50:
            # High-cardinality: frequency encoding
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0)
            encoders[col] = ("frequency", freq_map)
        else:
            # Low-cardinality: label encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = ("label", le)

    df["isFraud"] = target.values

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "preprocessed_data.csv")
    df.to_csv(out_path, index=False)

    # Save preprocessors
    with open(os.path.join(output_dir, "preprocessors.pkl"), "wb") as f:
        pickle.dump({"num_imputer": num_imputer, "encoders": encoders}, f)

    stats = {
        "output_shape": df.shape,
        "numeric_cols": len(numeric_cols),
        "cat_cols": len(cat_cols),
        "output_path": out_path,
    }
    print(f"\nPreprocessing complete. Shape: {df.shape}")
    return stats


if __name__ == "__main__":
    preprocessing(
        input_path="/mnt/e/sem 8/MLOPs/a4/artifacts/raw_data.csv",
        output_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
    )
