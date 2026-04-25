"""
Task 8: Intelligent Retraining Strategy
Compares: threshold-based vs periodic vs hybrid
Metrics: stability, compute cost, performance improvement
"""

import pandas as pd
import numpy as np
import pickle
import json
import time
import os
from pathlib import Path
from sklearn.metrics import recall_score, roc_auc_score, f1_score
from xgboost import XGBClassifier


ARTIFACTS_DIR = "/mnt/e/sem 8/MLOPs/a4/artifacts"
MODELS_DIR    = "/mnt/e/sem 8/MLOPs/a4/models"
OUTPUT_DIR    = "/mnt/e/sem 8/MLOPs/a4/artifacts/retraining"


def _evaluate_model(model, X, y):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "recall":  round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "auc":     round(float(roc_auc_score(y, y_proba)), 4),
        "f1":      round(float(f1_score(y, y_pred, zero_division=0)), 4),
    }


def _train_model(X_train, y_train, scale_pos_weight=8):
    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42,
        use_label_encoder=False, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def compare_retraining_strategies():
    """
    Strategy 1: Threshold-based  — retrain when recall < 0.70
    Strategy 2: Periodic         — retrain every N batches
    Strategy 3: Hybrid           — threshold + periodic combined
    """
    print("=" * 60)
    print("TASK 8: INTELLIGENT RETRAINING STRATEGY COMPARISON")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    # Simulate 5 data batches with increasing drift
    batches = []
    np.random.seed(42)
    for i in range(5):
        drift_factor = 1 + i * 0.15
        n = len(X_test) // 5
        X_b = X_test.iloc[i*n:(i+1)*n].copy()
        y_b = y_test.iloc[i*n:(i+1)*n].copy()
        # Inject drift into later batches
        if i >= 2 and "TransactionAmt" in X_b.columns:
            X_b["TransactionAmt"] = X_b["TransactionAmt"] * drift_factor
        batches.append((X_b, y_b))

    results = {}

    # ── Strategy 1: Threshold-Based ───────────────────────────────
    print("\n[Strategy 1] Threshold-Based (retrain if recall < 0.70)")
    with open(os.path.join(MODELS_DIR, "xgb_cost_sensitive.pkl"), "rb") as f:
        model = pickle.load(f)

    recalls, retrain_events, total_compute = [], [], 0
    for i, (X_b, y_b) in enumerate(batches):
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X_b.columns: X_b[col] = 0
            X_b = X_b[model.feature_names_in_]

        metrics = _evaluate_model(model, X_b, y_b)
        recalls.append(metrics["recall"])
        print(f"  Batch {i+1}: Recall={metrics['recall']:.4f}", end="")

        if metrics["recall"] < 0.70:
            t0 = time.time()
            X_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
            y_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train_raw.csv")).squeeze()
            model = _train_model(X_tr, y_tr)
            elapsed = time.time() - t0
            total_compute += elapsed
            retrain_events.append(i + 1)
            print(f"  ← RETRAINED ({elapsed:.1f}s)", end="")
        print()

    results["threshold_based"] = {
        "recalls":        recalls,
        "retrain_count":  len(retrain_events),
        "retrain_batches": retrain_events,
        "compute_seconds": round(total_compute, 2),
        "avg_recall":     round(float(np.mean(recalls)), 4),
        "stability":      round(float(1 - np.std(recalls)), 4),
    }

    # ── Strategy 2: Periodic ──────────────────────────────────────
    print("\n[Strategy 2] Periodic (retrain every 2 batches)")
    with open(os.path.join(MODELS_DIR, "xgb_cost_sensitive.pkl"), "rb") as f:
        model = pickle.load(f)

    recalls, retrain_events, total_compute = [], [], 0
    for i, (X_b, y_b) in enumerate(batches):
        if (i + 1) % 2 == 0:
            t0 = time.time()
            X_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
            y_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train_raw.csv")).squeeze()
            model = _train_model(X_tr, y_tr)
            elapsed = time.time() - t0
            total_compute += elapsed
            retrain_events.append(i + 1)
            print(f"  Batch {i+1}: RETRAINED ({elapsed:.1f}s)")

        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X_b.columns: X_b[col] = 0
            X_b = X_b[model.feature_names_in_]

        metrics = _evaluate_model(model, X_b, y_b)
        recalls.append(metrics["recall"])
        print(f"  Batch {i+1}: Recall={metrics['recall']:.4f}")

    results["periodic"] = {
        "recalls":         recalls,
        "retrain_count":   len(retrain_events),
        "retrain_batches": retrain_events,
        "compute_seconds": round(total_compute, 2),
        "avg_recall":      round(float(np.mean(recalls)), 4),
        "stability":       round(float(1 - np.std(recalls)), 4),
    }

    # ── Strategy 3: Hybrid ────────────────────────────────────────
    print("\n[Strategy 3] Hybrid (periodic every 3 + threshold 0.68)")
    with open(os.path.join(MODELS_DIR, "xgb_cost_sensitive.pkl"), "rb") as f:
        model = pickle.load(f)

    recalls, retrain_events, total_compute = [], [], 0
    for i, (X_b, y_b) in enumerate(batches):
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in X_b.columns: X_b[col] = 0
            X_b = X_b[model.feature_names_in_]

        metrics = _evaluate_model(model, X_b, y_b)
        recalls.append(metrics["recall"])

        should_retrain = (i + 1) % 3 == 0 or metrics["recall"] < 0.68
        if should_retrain:
            t0 = time.time()
            X_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train_raw.csv"))
            y_tr = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train_raw.csv")).squeeze()
            model = _train_model(X_tr, y_tr)
            elapsed = time.time() - t0
            total_compute += elapsed
            retrain_events.append(i + 1)
            reason = "periodic" if (i+1) % 3 == 0 else "threshold"
            print(f"  Batch {i+1}: Recall={metrics['recall']:.4f}  ← RETRAINED ({reason})")
        else:
            print(f"  Batch {i+1}: Recall={metrics['recall']:.4f}")

    results["hybrid"] = {
        "recalls":         recalls,
        "retrain_count":   len(retrain_events),
        "retrain_batches": retrain_events,
        "compute_seconds": round(total_compute, 2),
        "avg_recall":      round(float(np.mean(recalls)), 4),
        "stability":       round(float(1 - np.std(recalls)), 4),
    }

    # ── Summary Comparison ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print(f"{'Strategy':<20} {'Avg Recall':<12} {'Retrains':<10} {'Stability':<12} {'Compute(s)'}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<20} {r['avg_recall']:<12.4f} {r['retrain_count']:<10} {r['stability']:<12.4f} {r['compute_seconds']:.1f}s")

    with open(os.path.join(OUTPUT_DIR, "retraining_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/retraining_comparison.json")
    return results


if __name__ == "__main__":
    compare_retraining_strategies()
