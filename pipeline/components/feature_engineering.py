import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def feature_engineering(input_path: str, output_dir: str, imbalance_strategy: str = "smote") -> dict:
    """
    Feature engineering + imbalance handling.
    imbalance_strategy: 'smote', 'class_weight', or 'undersample'
    Compares SMOTE vs class_weight strategies.
    """
    print("=" * 50)
    print("STEP 4: FEATURE ENGINEERING")
    print("=" * 50)

    df = pd.read_csv(input_path)

    # New interaction features
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_zscore"] = (
            df["TransactionAmt"] - df["TransactionAmt"].mean()
        ) / (df["TransactionAmt"].std() + 1e-9)

    # Hour/day features from TransactionDT
    if "TransactionDT" in df.columns:
        df["hour"] = (df["TransactionDT"] / 3600 % 24).astype(int)
        df["day_of_week"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
        df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    X = df.drop(columns=["isFraud"], errors="ignore")
    y = df["isFraud"]

    print(f"Before imbalance handling: {Counter(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Strategy 1: SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:        {Counter(y_train_smote)}")

    # Strategy 2: Random Undersampling
    rus = RandomUnderSampler(random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
    print(f"After Undersampling:{Counter(y_train_under)}")

    # Strategy 3: class_weight (no resampling — passed to model)
    class_ratio = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Class weight ratio: {class_ratio} (used as scale_pos_weight)")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save all splits
    X_train_smote.to_csv(os.path.join(output_dir, "X_train_smote.csv"), index=False)
    y_train_smote.to_csv(os.path.join(output_dir, "y_train_smote.csv"), index=False)
    X_train_under.to_csv(os.path.join(output_dir, "X_train_under.csv"), index=False)
    y_train_under.to_csv(os.path.join(output_dir, "y_train_under.csv"), index=False)
    X_train.to_csv(os.path.join(output_dir, "X_train_raw.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train_raw.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    meta = {"class_weight_ratio": class_ratio, "output_dir": output_dir}
    with open(os.path.join(output_dir, "fe_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    stats = {
        "train_smote_shape": X_train_smote.shape,
        "train_under_shape": X_train_under.shape,
        "test_shape": X_test.shape,
        "class_weight_ratio": class_ratio,
    }
    print(f"\nFeature engineering complete: {stats}")
    return stats


if __name__ == "__main__":
    feature_engineering(
        input_path="/mnt/e/sem 8/MLOPs/a4/artifacts/preprocessed_data.csv",
        output_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
    )
