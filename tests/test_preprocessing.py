import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def raw_df():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "TransactionID":  range(n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "TransactionAmt": np.random.exponential(100, n),
        "card1":          np.random.randint(1000, 9999, n).astype(float),
        "ProductCD":      np.random.choice(["W", "H", None], n),
        "sparse_col":     [None] * n,
    })
    df.loc[::10, "TransactionAmt"] = np.nan
    return df


def test_sparse_columns_dropped(raw_df):
    thresh = int(0.1 * len(raw_df))
    result = raw_df.dropna(thresh=thresh, axis=1)
    assert "sparse_col" not in result.columns


def test_no_nulls_after_numeric_imputation(raw_df):
    from sklearn.impute import SimpleImputer
    df = raw_df.drop(columns=["sparse_col", "ProductCD"])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]
    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])
    assert df[num_cols].isnull().sum().sum() == 0


def test_label_encoding_no_nulls(raw_df):
    from sklearn.preprocessing import LabelEncoder
    col = raw_df["ProductCD"].fillna("MISSING")
    le = LabelEncoder()
    encoded = le.fit_transform(col.astype(str))
    assert len(encoded) == len(raw_df)
    assert not pd.isna(encoded).any()


def test_feature_engineering_adds_columns(raw_df):
    df = raw_df.copy().fillna(0)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    assert "TransactionAmt_log" in df.columns
    assert (df["TransactionAmt_log"] >= 0).all()
