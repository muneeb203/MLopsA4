import pandas as pd
import json
import os
from pathlib import Path


SCHEMA = {
    "required_columns": ["TransactionID", "isFraud", "TransactionAmt", "ProductCD"],
    "target_column": "isFraud",
    "numeric_ranges": {
        "TransactionAmt": (0, 1e6),
        "isFraud": (0, 1),
    },
}


def data_validation(input_path: str, output_dir: str) -> dict:
    """Validate schema, missing values, and data quality."""
    print("=" * 50)
    print("STEP 2: DATA VALIDATION")
    print("=" * 50)

    df = pd.read_csv(input_path)
    report = {"passed": True, "checks": {}, "warnings": []}

    # Check 1: Required columns
    missing_cols = [c for c in SCHEMA["required_columns"] if c not in df.columns]
    report["checks"]["required_columns"] = {
        "passed": len(missing_cols) == 0,
        "missing": missing_cols,
    }
    if missing_cols:
        report["passed"] = False
        print(f"FAIL: Missing columns: {missing_cols}")
    else:
        print("PASS: All required columns present")

    # Check 2: Missing value analysis
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing = missing_pct[missing_pct > 80].to_dict()
    report["checks"]["missing_values"] = {
        "total_missing_cols": int((df.isnull().sum() > 0).sum()),
        "high_missing_cols": high_missing,
        "passed": True,
    }
    print(f"INFO: {len(high_missing)} columns with >80% missing values")
    if high_missing:
        report["warnings"].append(f"High missing: {list(high_missing.keys())[:5]}")

    # Check 3: Target distribution
    fraud_rate = df["isFraud"].mean()
    report["checks"]["target_distribution"] = {
        "fraud_rate": float(fraud_rate),
        "fraud_count": int(df["isFraud"].sum()),
        "passed": 0.001 < fraud_rate < 0.5,
    }
    print(f"INFO: Fraud rate = {fraud_rate:.4f}")

    # Check 4: Numeric range validation
    for col, (lo, hi) in SCHEMA["numeric_ranges"].items():
        if col in df.columns:
            out_of_range = int(((df[col] < lo) | (df[col] > hi)).sum())
            report["checks"][f"range_{col}"] = {
                "passed": out_of_range == 0,
                "out_of_range_count": out_of_range,
            }

    # Check 5: Duplicate TransactionIDs
    dupes = int(df["TransactionID"].duplicated().sum())
    report["checks"]["duplicates"] = {"passed": dupes == 0, "count": dupes}
    print(f"INFO: Duplicate TransactionIDs = {dupes}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = os.path.join(output_dir, "validation_report.json")

    def _to_native(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: _to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_native(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(report_path, "w") as f:
        json.dump(_to_native(report), f, indent=2)

    status = "PASSED" if report["passed"] else "FAILED"
    print(f"\nValidation {status}. Report saved to {report_path}")

    if not report["passed"]:
        raise ValueError(f"Data validation failed: {report['checks']}")

    return report


if __name__ == "__main__":
    report = data_validation(
        input_path="/mnt/e/sem 8/MLOPs/a4/artifacts/raw_data.csv",
        output_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
    )
