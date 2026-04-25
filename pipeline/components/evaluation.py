import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve,
)


def evaluation(artifacts_dir: str, models_dir: str, output_dir: str) -> dict:
    """
    Evaluate all models on test set.
    Reports: Precision, Recall, F1, AUC-ROC, Confusion Matrix.
    Compares standard vs cost-sensitive training.
    """
    print("=" * 50)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 50)

    X_test = pd.read_csv(os.path.join(artifacts_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(artifacts_dir, "y_test.csv")).squeeze()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_names = ["xgb_standard", "xgb_cost_sensitive", "lgbm", "rf_hybrid"]
    all_results = {}

    for name in model_names:
        model_path = os.path.join(models_dir, f"{name}.pkl")
        if not os.path.exists(model_path):
            print(f"  Skipping {name} (not found)")
            continue

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_proba)
        cm        = confusion_matrix(y_test, y_pred).tolist()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Cost-sensitive business impact
        fraud_loss_per_fn = 500   # avg fraud amount ($)
        false_alarm_cost  = 10    # investigation cost ($)
        total_cost = fn * fraud_loss_per_fn + fp * false_alarm_cost

        result = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1_score":  round(f1, 4),
            "auc_roc":   round(auc, 4),
            "fpr":       round(fpr, 4),
            "confusion_matrix": cm,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "business_cost_usd": int(total_cost),
        }
        all_results[name] = result

        print(f"\n{'─'*40}")
        print(f"Model: {name}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}  ← KEY METRIC")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  AUC-ROC   : {auc:.4f}")
        print(f"  FPR       : {fpr:.4f}")
        print(f"  Business $ : ${total_cost:,}")

        # Confusion matrix plot
        _plot_confusion_matrix(cm, name, output_dir)

    # Compare standard vs cost-sensitive
    if "xgb_standard" in all_results and "xgb_cost_sensitive" in all_results:
        std = all_results["xgb_standard"]
        cs  = all_results["xgb_cost_sensitive"]
        comparison = {
            "recall_improvement": round(cs["recall"] - std["recall"], 4),
            "cost_reduction_usd": std["business_cost_usd"] - cs["business_cost_usd"],
            "precision_tradeoff": round(std["precision"] - cs["precision"], 4),
        }
        all_results["standard_vs_cost_sensitive_comparison"] = comparison
        print(f"\n{'='*50}")
        print("COMPARISON: Standard vs Cost-Sensitive XGBoost")
        print(f"  Recall improvement : +{comparison['recall_improvement']:.4f}")
        print(f"  Cost reduction     : ${comparison['cost_reduction_usd']:,}")
        print(f"  Precision tradeoff : -{comparison['precision_tradeoff']:.4f}")

    # Best model by recall (most important for fraud)
    scored = {k: v["recall"] for k, v in all_results.items() if "recall" in v}
    best_model = max(scored, key=scored.get)
    all_results["best_model"] = best_model
    print(f"\nBest model by Recall: {best_model} ({scored[best_model]:.4f})")

    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def _plot_confusion_matrix(cm, model_name, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {model_name}")
    plt.colorbar(im, ax=ax)
    classes = ["Legit", "Fraud"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > max(max(cm)) / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{model_name}.png"), dpi=100)
    plt.close()


if __name__ == "__main__":
    evaluation(
        artifacts_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
        models_dir="/mnt/e/sem 8/MLOPs/a4/models",
        output_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
    )
