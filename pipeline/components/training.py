import pandas as pd
import numpy as np
import pickle
import os
import json
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline


def training(artifacts_dir: str, output_dir: str) -> dict:
    """
    Train 3 models:
    1. XGBoost (with SMOTE data + cost-sensitive)
    2. LightGBM (with SMOTE data + cost-sensitive)
    3. RF + SelectFromModel hybrid (with undersampled data)
    Also compares standard vs cost-sensitive training.
    """
    print("=" * 50)
    print("STEP 5: MODEL TRAINING")
    print("=" * 50)

    # Load data
    X_train_smote = pd.read_csv(os.path.join(artifacts_dir, "X_train_smote.csv"))
    y_train_smote = pd.read_csv(os.path.join(artifacts_dir, "y_train_smote.csv")).squeeze()
    X_train_raw   = pd.read_csv(os.path.join(artifacts_dir, "X_train_raw.csv"))
    y_train_raw   = pd.read_csv(os.path.join(artifacts_dir, "y_train_raw.csv")).squeeze()
    X_train_under = pd.read_csv(os.path.join(artifacts_dir, "X_train_under.csv"))
    y_train_under = pd.read_csv(os.path.join(artifacts_dir, "y_train_under.csv")).squeeze()

    with open(os.path.join(artifacts_dir, "fe_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    class_weight_ratio = meta["class_weight_ratio"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_summary = {}

    # ── Model 1: XGBoost standard (SMOTE) ──────────────────────────
    print("\n[1/4] XGBoost Standard (SMOTE)...")
    xgb_std = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", random_state=42,
        use_label_encoder=False, verbosity=0,
    )
    xgb_std.fit(X_train_smote, y_train_smote)
    _save_model(xgb_std, output_dir, "xgb_standard")
    training_summary["xgb_standard"] = {"strategy": "smote", "cost_sensitive": False}

    # ── Model 2: XGBoost cost-sensitive (scale_pos_weight) ─────────
    print("[2/4] XGBoost Cost-Sensitive (class_weight)...")
    xgb_cs = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=class_weight_ratio,
        eval_metric="auc", random_state=42,
        use_label_encoder=False, verbosity=0,
    )
    xgb_cs.fit(X_train_raw, y_train_raw)
    _save_model(xgb_cs, output_dir, "xgb_cost_sensitive")
    training_summary["xgb_cost_sensitive"] = {
        "strategy": "class_weight",
        "cost_sensitive": True,
        "scale_pos_weight": class_weight_ratio,
    }

    # ── Model 3: LightGBM cost-sensitive ───────────────────────────
    print("[3/4] LightGBM Cost-Sensitive...")
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42, verbose=-1,
    )
    lgbm.fit(X_train_smote, y_train_smote)
    _save_model(lgbm, output_dir, "lgbm")
    training_summary["lgbm"] = {"strategy": "smote", "cost_sensitive": True}

    # ── Model 4: RF + SelectFromModel hybrid ───────────────────────
    print("[4/4] Random Forest + Feature Selection Hybrid...")
    rf_pipeline = Pipeline([
        ("selector", SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced"),
            threshold="mean",
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=100, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    rf_pipeline.fit(X_train_under, y_train_under)
    _save_model(rf_pipeline, output_dir, "rf_hybrid")
    training_summary["rf_hybrid"] = {"strategy": "undersample", "cost_sensitive": True}

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"\nTraining complete. Models saved to {output_dir}")
    return training_summary


def _save_model(model, output_dir, name):
    path = os.path.join(output_dir, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved: {name}.pkl")


if __name__ == "__main__":
    training(
        artifacts_dir="/mnt/e/sem 8/MLOPs/a4/artifacts",
        output_dir="/mnt/e/sem 8/MLOPs/a4/models",
    )
