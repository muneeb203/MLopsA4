"""
Task 7: Realistic Drift Simulation
- Time-based drift: train on earlier transactions, test on later
- New fraud patterns introduced in later data
- Feature importance shifts measured
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from pathlib import Path


ARTIFACTS_DIR = "/mnt/e/sem 8/MLOPs/a4/artifacts"
MODELS_DIR    = "/mnt/e/sem 8/MLOPs/a4/models"
OUTPUT_DIR    = "/mnt/e/sem 8/MLOPs/a4/artifacts/drift"


def simulate_time_based_drift():
    """
    Simulate realistic temporal drift:
    1. Split data chronologically (early vs late transactions)
    2. Introduce new fraud patterns in late data
    3. Measure feature distribution shifts (KS test)
    4. Show impact on model performance
    """
    print("=" * 60)
    print("TASK 7: DRIFT SIMULATION (Time-Based)")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(os.path.join(ARTIFACTS_DIR, "preprocessed_data.csv"))

    # ── Temporal Split ────────────────────────────────────────────
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT")
        split_idx = int(len(df) * 0.7)
        df_early = df.iloc[:split_idx].copy()
        df_late  = df.iloc[split_idx:].copy()
    else:
        split_idx = int(len(df) * 0.7)
        df_early  = df.iloc[:split_idx].copy()
        df_late   = df.iloc[split_idx:].copy()

    print(f"Early period: {len(df_early):,} rows  | Fraud rate: {df_early['isFraud'].mean():.4f}")
    print(f"Late period:  {len(df_late):,} rows   | Fraud rate: {df_late['isFraud'].mean():.4f}")

    # ── Inject New Fraud Patterns into Late Data ──────────────────
    late_legit = df_late[df_late["isFraud"] == 0].copy()
    n_new_fraud = int(len(late_legit) * 0.03)

    new_fraud = late_legit.sample(n=min(n_new_fraud, len(late_legit)), random_state=42).copy()
    new_fraud["isFraud"] = 1

    # New pattern: high-value night transactions
    if "TransactionAmt" in new_fraud.columns:
        new_fraud["TransactionAmt"] = new_fraud["TransactionAmt"] * np.random.uniform(3, 10, len(new_fraud))
    if "is_night" in new_fraud.columns:
        new_fraud["is_night"] = 1

    df_late_drifted = pd.concat([df_late, new_fraud]).sample(frac=1, random_state=42)
    print(f"\nAfter drift injection:")
    print(f"  Late fraud rate: {df_late_drifted['isFraud'].mean():.4f} (was {df_late['isFraud'].mean():.4f})")

    # ── KS Test for Feature Drift ─────────────────────────────────
    numeric_cols = df_early.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "isFraud"][:20]

    drift_results = {}
    for col in numeric_cols:
        stat, pval = ks_2samp(df_early[col].dropna(), df_late_drifted[col].dropna())
        drift_results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value":      round(float(pval), 6),
            "drifted":      pval < 0.05,
        }

    drifted_features = [k for k, v in drift_results.items() if v["drifted"]]
    drift_score = len(drifted_features) / len(numeric_cols)
    print(f"\nDrift detected in {len(drifted_features)}/{len(numeric_cols)} features")
    print(f"Overall drift score: {drift_score:.4f}")

    # ── Model Performance Before vs After Drift ───────────────────
    model_path = os.path.join(MODELS_DIR, "xgb_cost_sensitive.pkl")
    perf_comparison = {}

    if os.path.exists(model_path):
        from sklearn.metrics import recall_score, roc_auc_score

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        for label, data in [("early (no drift)", df_early), ("late (with drift)", df_late_drifted)]:
            X = data.drop(columns=["isFraud"], errors="ignore")
            y = data["isFraud"]

            if hasattr(model, "feature_names_in_"):
                for col in model.feature_names_in_:
                    if col not in X.columns:
                        X[col] = 0
                X = X[model.feature_names_in_]

            try:
                recall = recall_score(y, model.predict(X))
                auc    = roc_auc_score(y, model.predict_proba(X)[:, 1])
                perf_comparison[label] = {"recall": round(recall, 4), "auc": round(auc, 4)}
                print(f"  [{label}] Recall={recall:.4f}  AUC={auc:.4f}")
            except Exception as e:
                print(f"  [{label}] Error: {e}")

    # ── Plot Top Drifted Features ─────────────────────────────────
    top_drifted = sorted(drift_results.items(), key=lambda x: x[1]["ks_statistic"], reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (col, stats) in enumerate(top_drifted):
        axes[i].hist(df_early[col].dropna(), bins=30, alpha=0.6, label="Early", color="blue", density=True)
        axes[i].hist(df_late_drifted[col].dropna(), bins=30, alpha=0.6, label="Late+Drift", color="red", density=True)
        axes[i].set_title(f"{col}\nKS={stats['ks_statistic']:.3f} p={stats['p_value']:.4f}")
        axes[i].legend(fontsize=8)

    plt.suptitle("Feature Distribution Drift: Early vs Late Transactions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "drift_visualization.png"), dpi=100)
    plt.close()
    print(f"\nDrift visualization saved.")

    summary = {
        "drift_score":       round(drift_score, 4),
        "drifted_features":  drifted_features[:10],
        "total_features":    len(numeric_cols),
        "performance_impact": perf_comparison,
        "new_fraud_injected": n_new_fraud,
    }

    with open(os.path.join(OUTPUT_DIR, "drift_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDrift report saved to {OUTPUT_DIR}/drift_report.json")
    return summary


if __name__ == "__main__":
    result = simulate_time_based_drift()
    print("\n" + "=" * 60)
    print(f"Drift Score: {result['drift_score']:.4f}")
    print(f"Features drifted: {len(result['drifted_features'])}")
