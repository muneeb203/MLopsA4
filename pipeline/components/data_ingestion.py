import pandas as pd
import os
import json
from pathlib import Path


def data_ingestion(data_dir: str, output_dir: str, sample_frac: float = 0.3) -> dict:
    """Load, merge and sample the IEEE CIS fraud dataset."""
    print("=" * 50)
    print("STEP 1: DATA INGESTION")
    print("=" * 50)

    train_transaction = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    train_identity = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))

    print(f"Transaction shape: {train_transaction.shape}")
    print(f"Identity shape:    {train_identity.shape}")

    df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    print(f"Merged shape:      {df.shape}")

    # Stratified sample to preserve fraud ratio
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(
        frac=sample_frac, random_state=42
    )
    df_sampled = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Sampled shape:     {df_sampled.shape}")
    print(f"Fraud rate:        {df_sampled['isFraud'].mean():.4f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, "raw_data.csv")
    df_sampled.to_csv(out_path, index=False)

    stats = {
        "rows": int(df_sampled.shape[0]),
        "cols": int(df_sampled.shape[1]),
        "fraud_rate": float(df_sampled["isFraud"].mean()),
        "fraud_count": int(df_sampled["isFraud"].sum()),
        "output_path": out_path,
    }
    print(f"\nIngestion stats: {json.dumps(stats, indent=2)}")
    return stats


if __name__ == "__main__":
    stats = data_ingestion(
        data_dir="/mnt/e/sem 8/MLOPs/a4/data",
        output_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
    )
    print("\nData ingestion complete.")
