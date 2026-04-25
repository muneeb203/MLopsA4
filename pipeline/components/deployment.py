import json
import os
import pickle
import shutil
from pathlib import Path


RECALL_THRESHOLD = 0.70
AUC_THRESHOLD    = 0.85


def deployment(eval_results_path: str, models_dir: str, deploy_dir: str) -> dict:
    """
    Conditional deployment:
    - Deploy only if best model recall > RECALL_THRESHOLD AND AUC > AUC_THRESHOLD
    - Copies best model to deploy_dir as 'production_model.pkl'
    """
    print("=" * 50)
    print("STEP 7: CONDITIONAL DEPLOYMENT")
    print("=" * 50)

    with open(eval_results_path, "r") as f:
        results = json.load(f)

    best_model = results.get("best_model", "xgb_cost_sensitive")
    metrics    = results.get(best_model, {})

    recall  = metrics.get("recall", 0)
    auc_roc = metrics.get("auc_roc", 0)

    print(f"Best model   : {best_model}")
    print(f"Recall       : {recall:.4f}  (threshold: {RECALL_THRESHOLD})")
    print(f"AUC-ROC      : {auc_roc:.4f}  (threshold: {AUC_THRESHOLD})")

    deploy_decision = {
        "model_name": best_model,
        "recall": recall,
        "auc_roc": auc_roc,
        "recall_threshold": RECALL_THRESHOLD,
        "auc_threshold": AUC_THRESHOLD,
    }

    if recall >= RECALL_THRESHOLD and auc_roc >= AUC_THRESHOLD:
        Path(deploy_dir).mkdir(parents=True, exist_ok=True)
        src = os.path.join(models_dir, f"{best_model}.pkl")
        dst = os.path.join(deploy_dir, "production_model.pkl")
        shutil.copy(src, dst)

        deploy_decision["deployed"] = True
        deploy_decision["deploy_path"] = dst
        print(f"\nDEPLOYMENT APPROVED: Model deployed to {dst}")
    else:
        reason = []
        if recall < RECALL_THRESHOLD:
            reason.append(f"Recall {recall:.4f} < {RECALL_THRESHOLD}")
        if auc_roc < AUC_THRESHOLD:
            reason.append(f"AUC {auc_roc:.4f} < {AUC_THRESHOLD}")

        deploy_decision["deployed"] = False
        deploy_decision["reason"] = "; ".join(reason)
        print(f"\nDEPLOYMENT REJECTED: {deploy_decision['reason']}")
        print("Pipeline will retry with different hyperparameters.")

    result_path = os.path.join(deploy_dir if deploy_decision["deployed"] else models_dir,
                               "deploy_decision.json")
    with open(result_path, "w") as f:
        json.dump(deploy_decision, f, indent=2)

    return deploy_decision


if __name__ == "__main__":
    result = deployment(
        eval_results_path="/mnt/e/sem 8/MLOPs/a4/artifacts/evaluation_results.json",
        models_dir="/mnt/e/sem 8/MLOPs/a4/models",
        deploy_dir="/mnt/e/sem 8/MLOPs/a4/models/production",
    )
    print(f"\nDeployed: {result['deployed']}")
