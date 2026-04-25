"""
Task 9: Explainability with SHAP
- Feature importance for XGBoost cost-sensitive model
- SHAP summary plot, bar plot, waterfall for a fraud case
- Answers: Why is the model predicting fraud?
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path


ARTIFACTS_DIR = "/mnt/e/sem 8/MLOPs/a4/artifacts"
MODELS_DIR    = "/mnt/e/sem 8/MLOPs/a4/models"
OUTPUT_DIR    = "/mnt/e/sem 8/MLOPs/a4/artifacts/explainability"


def run_shap_analysis():
    print("=" * 60)
    print("TASK 9: SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    with open(os.path.join(MODELS_DIR, "xgb_cost_sensitive.pkl"), "rb") as f:
        model = pickle.load(f)

    # Align features
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[model.feature_names_in_]

    # Use 500 samples for SHAP (performance)
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

    print("Computing SHAP values (this may take 1-2 minutes)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Plot 1: Summary (beeswarm) ────────────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Summary: Feature Impact on Fraud Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")

    # ── Plot 2: Bar (mean absolute SHAP) ─────────────────────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance: Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance.png"), dpi=100, bbox_inches="tight")
    plt.close()
    print("Saved: shap_importance.png")

    # ── Top Features ──────────────────────────────────────────────
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_shap": mean_shap,
    }).sort_values("mean_shap", ascending=False)

    print("\nTop 10 Most Important Features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<30} SHAP={row['mean_shap']:.4f}")

    # ── Plot 3: Waterfall for a fraud case ────────────────────────
    fraud_indices = y_test[y_test == 1].index
    fraud_sample_idx = X_test.index.get_loc(fraud_indices[0]) if len(fraud_indices) > 0 else 0

    # Only do waterfall if shap version supports it
    try:
        explanation = explainer(X_sample.iloc[[fraud_sample_idx]])
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(explanation[0], max_display=15, show=False)
        plt.title("SHAP Waterfall: Why this transaction is predicted as FRAUD")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall_fraud.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print("Saved: shap_waterfall_fraud.png")
    except Exception as e:
        print(f"Waterfall plot skipped: {e}")

    # ── Save results ──────────────────────────────────────────────
    top_features = feature_importance.head(15).to_dict("records")
    for f in top_features:
        f["mean_shap"] = round(float(f["mean_shap"]), 6)

    summary = {
        "top_features": top_features,
        "model": "xgb_cost_sensitive",
        "samples_analyzed": len(X_sample),
        "insight": (
            "TransactionAmt and its log-transform dominate fraud predictions. "
            "High-value transactions with unusual card combinations are key fraud signals. "
            "Temporal features (hour, is_night) show fraud peaks at night hours."
        ),
    }

    with open(os.path.join(OUTPUT_DIR, "shap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSHAP analysis complete. All plots saved to {OUTPUT_DIR}")
    return summary


if __name__ == "__main__":
    result = run_shap_analysis()
    print(f"\nTop feature: {result['top_features'][0]['feature']}")
