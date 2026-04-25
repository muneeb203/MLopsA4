"""
Kubeflow Pipelines v2 — Fraud Detection Pipeline
7 steps with conditional deployment and retry mechanisms.
"""

import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics
import os

PIPELINE_IMAGE = "muneebqureshi03/fraud-pipeline:latest"
DATA_DIR       = "/mnt/e/sem 8/MLOPs/a4/data"
ARTIFACTS_DIR  = "/mnt/e/sem 8/MLOPs/a4/artifacts"
MODELS_DIR     = "/mnt/e/sem 8/MLOPs/a4/models"


# ── Component 1: Data Ingestion ─────────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def data_ingestion_op(
    data_dir: str,
    sample_frac: float,
    output_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    import os

    train_t = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    train_i = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    df = train_t.merge(train_i, on="TransactionID", how="left")

    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0].sample(frac=sample_frac, random_state=42)
    df_s = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_s.to_csv(output_dataset.path, index=False)
    print(f"Ingested {df_s.shape[0]} rows. Fraud rate: {df_s['isFraud'].mean():.4f}")


# ── Component 2: Data Validation ────────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def data_validation_op(
    input_dataset: Input[Dataset],
    validation_metrics: Output[Metrics],
) -> bool:
    import pandas as pd
    import json

    df = pd.read_csv(input_dataset.path)
    required = ["TransactionID", "isFraud", "TransactionAmt", "ProductCD"]
    missing  = [c for c in required if c not in df.columns]

    fraud_rate = float(df["isFraud"].mean())
    dupes      = int(df["TransactionID"].duplicated().sum())
    high_miss  = int((df.isnull().mean() > 0.8).sum())

    validation_metrics.log_metric("fraud_rate", fraud_rate)
    validation_metrics.log_metric("duplicate_ids", dupes)
    validation_metrics.log_metric("high_missing_cols", high_miss)

    passed = len(missing) == 0 and dupes == 0
    print(f"Validation passed: {passed}")
    return passed


# ── Component 3: Preprocessing ──────────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def preprocessing_op(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(input_dataset.path)
    thresh = int(0.1 * len(df))
    df = df.dropna(thresh=thresh, axis=1)

    target = df["isFraud"].copy()
    df = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    imp = SimpleImputer(strategy="median")
    df[num_cols] = imp.fit_transform(df[num_cols])

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")
        if df[col].nunique() > 50:
            freq = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq).fillna(0)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df["isFraud"] = target.values
    df.to_csv(output_dataset.path, index=False)
    print(f"Preprocessed shape: {df.shape}")


# ── Component 4: Feature Engineering ────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def feature_engineering_op(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv(input_dataset.path)

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"]    = np.log1p(df["TransactionAmt"])
        df["TransactionAmt_zscore"] = (df["TransactionAmt"] - df["TransactionAmt"].mean()) / (df["TransactionAmt"].std() + 1e-9)
    if "TransactionDT" in df.columns:
        df["hour"]        = (df["TransactionDT"] / 3600 % 24).astype(int)
        df["day_of_week"] = (df["TransactionDT"] / (3600 * 24) % 7).astype(int)
        df["is_night"]    = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_tr_s, y_tr_s = smote.fit_resample(X_tr, y_tr)

    train_df = pd.concat([X_tr_s, y_tr_s], axis=1)
    test_df  = pd.concat([X_te, y_te], axis=1)
    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")


# ── Component 5: Model Training ──────────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def training_op(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
):
    import pandas as pd
    import pickle
    from xgboost import XGBClassifier

    df     = pd.read_csv(train_dataset.path)
    X      = df.drop(columns=["isFraud"])
    y      = df["isFraud"]

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=8, eval_metric="auc",
        random_state=42, use_label_encoder=False, verbosity=0,
    )
    model.fit(X, y)

    with open(model_artifact.path + ".pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved.")


# ── Component 6: Evaluation ──────────────────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def evaluation_op(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics],
) -> float:
    import pandas as pd
    import pickle
    from sklearn.metrics import recall_score, roc_auc_score, f1_score

    df    = pd.read_csv(test_dataset.path)
    X     = df.drop(columns=["isFraud"])
    y     = df["isFraud"]

    with open(model_artifact.path + ".pkl", "rb") as f:
        model = pickle.load(f)

    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    recall  = float(recall_score(y, y_pred))
    auc     = float(roc_auc_score(y, y_proba))
    f1      = float(f1_score(y, y_pred))

    eval_metrics.log_metric("recall",  recall)
    eval_metrics.log_metric("auc_roc", auc)
    eval_metrics.log_metric("f1",      f1)

    print(f"Recall={recall:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
    return recall


# ── Component 7: Conditional Deployment ─────────────────────────────────────
@component(base_image=PIPELINE_IMAGE, packages_to_install=[])
def deployment_op(
    model_artifact: Input[Model],
    recall: float,
    recall_threshold: float = 0.70,
):
    import shutil, os

    if recall >= recall_threshold:
        deploy_path = "/tmp/production_model.pkl"
        shutil.copy(model_artifact.path + ".pkl", deploy_path)
        print(f"DEPLOYED: recall={recall:.4f} >= threshold={recall_threshold}")
    else:
        print(f"REJECTED: recall={recall:.4f} < threshold={recall_threshold}")
        raise ValueError(f"Model recall {recall:.4f} below threshold {recall_threshold}")


# ── Pipeline Definition ───────────────────────────────────────────────────────
@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="IEEE CIS Fraud Detection — 7-step KFP pipeline with conditional deployment",
)
def fraud_detection_pipeline(
    data_dir:        str   = DATA_DIR,
    sample_frac:     float = 0.3,
    recall_threshold: float = 0.70,
):
    # Step 1 — Ingest
    ingest = data_ingestion_op(
        data_dir=data_dir,
        sample_frac=sample_frac,
    ).set_retry(num_retries=2, backoff_duration="30s")

    # Step 2 — Validate
    validate = data_validation_op(
        input_dataset=ingest.outputs["output_dataset"],
    ).set_retry(num_retries=1)

    # Step 3 — Preprocess
    preprocess = preprocessing_op(
        input_dataset=ingest.outputs["output_dataset"],
    ).after(validate).set_retry(num_retries=2)

    # Step 4 — Feature Engineering
    feat_eng = feature_engineering_op(
        input_dataset=preprocess.outputs["output_dataset"],
    ).set_retry(num_retries=1)

    # Step 5 — Train
    train = training_op(
        train_dataset=feat_eng.outputs["train_dataset"],
    ).set_cpu_limit("2").set_memory_limit("4G").set_retry(num_retries=2)

    # Step 6 — Evaluate
    evaluate = evaluation_op(
        test_dataset=feat_eng.outputs["test_dataset"],
        model_artifact=train.outputs["model_artifact"],
    ).set_retry(num_retries=1)

    # Step 7 — Conditional Deploy (output named 'Output' for return value in KFP v2)
    with dsl.If(evaluate.outputs["Output"] >= recall_threshold, name="recall-check"):
        deployment_op(
            model_artifact=train.outputs["model_artifact"],
            recall=evaluate.outputs["Output"],
            recall_threshold=recall_threshold,
        )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.yaml",
    )
    print("Pipeline compiled to fraud_detection_pipeline.yaml")
