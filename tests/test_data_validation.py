import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "TransactionID":  range(n),
        "isFraud":        np.random.choice([0, 1], n, p=[0.95, 0.05]),
        "TransactionAmt": np.random.exponential(100, n),
        "ProductCD":      np.random.choice(["W", "H", "C"], n),
    })


def test_required_columns_present(sample_df):
    required = ["TransactionID", "isFraud", "TransactionAmt", "ProductCD"]
    for col in required:
        assert col in sample_df.columns, f"Missing column: {col}"


def test_fraud_rate_in_range(sample_df):
    rate = sample_df["isFraud"].mean()
    assert 0.001 < rate < 0.5, f"Fraud rate {rate} out of expected range"


def test_no_duplicate_transaction_ids(sample_df):
    dupes = sample_df["TransactionID"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate TransactionIDs"


def test_transaction_amount_positive(sample_df):
    assert (sample_df["TransactionAmt"] >= 0).all(), "Negative TransactionAmt found"


def test_target_binary(sample_df):
    unique_vals = set(sample_df["isFraud"].unique())
    assert unique_vals.issubset({0, 1}), f"Non-binary target values: {unique_vals}"
