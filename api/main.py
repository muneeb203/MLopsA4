"""
Fraud Detection Inference API
FastAPI with Prometheus metrics endpoint.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)
from fastapi.responses import Response
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["prediction"],
)
PREDICTION_CONFIDENCE = Histogram(
    "fraud_prediction_confidence",
    "Distribution of fraud prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_RECALL_GAUGE = Gauge(
    "fraud_model_recall",
    "Current model recall on recent predictions",
)
FALSE_POSITIVE_RATE_GAUGE = Gauge(
    "fraud_false_positive_rate",
    "Current false positive rate",
)
DATA_DRIFT_GAUGE = Gauge(
    "fraud_data_drift_score",
    "Current data drift score (0=no drift, 1=max drift)",
)
ACTIVE_MODEL = Gauge(
    "fraud_active_model_info",
    "Info about the currently active model",
    ["model_name", "version"],
)

# ── Load Model ────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "/mnt/e/sem 8/MLOPs/a4/models/production/production_model.pkl")
model       = None
model_name  = "xgb_cost_sensitive"

@app.on_event("startup")
def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        ACTIVE_MODEL.labels(model_name=model_name, version="1.0").set(1)
        MODEL_RECALL_GAUGE.set(0.8125)
        FALSE_POSITIVE_RATE_GAUGE.set(0.101)
        DATA_DRIFT_GAUGE.set(0.0)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")


# ── Request / Response Schemas ────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    TransactionAmt:   float
    ProductCD:        Optional[str] = "W"
    card1:            Optional[float] = 0
    card2:            Optional[float] = 0
    card3:            Optional[float] = 0
    card4:            Optional[str] = "visa"
    card5:            Optional[float] = 0
    card6:            Optional[str] = "debit"
    addr1:            Optional[float] = 0
    addr2:            Optional[float] = 0
    dist1:            Optional[float] = 0
    P_emaildomain:    Optional[str] = "gmail.com"
    R_emaildomain:    Optional[str] = "gmail.com"

class PredictionResponse(BaseModel):
    transaction_id:   Optional[str]
    is_fraud:         bool
    fraud_probability: float
    confidence:       str
    model_version:    str = "1.0"


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    start = time.time()

    if model is None:
        REQUEST_COUNT.labels("POST", "/predict", "500").inc()
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Build feature vector — fill missing features with 0
        features = pd.DataFrame([{
            "TransactionAmt":         transaction.TransactionAmt,
            "TransactionAmt_log":     np.log1p(transaction.TransactionAmt),
            "TransactionAmt_zscore":  0.0,
            "card1": transaction.card1, "card2": transaction.card2,
            "card3": transaction.card3, "card5": transaction.card5,
            "addr1": transaction.addr1, "addr2": transaction.addr2,
            "dist1": transaction.dist1,
        }])

        # Align with model's expected features
        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in features.columns:
                    features[col] = 0
            features = features[model.feature_names_in_]

        prob      = float(model.predict_proba(features)[0][1])
        is_fraud  = prob >= 0.5

        # Update Prometheus metrics
        FRAUD_PREDICTIONS.labels(prediction="fraud" if is_fraud else "legit").inc()
        PREDICTION_CONFIDENCE.observe(prob)
        REQUEST_COUNT.labels("POST", "/predict", "200").inc()

        confidence = "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.1 else "low"

        latency = time.time() - start
        REQUEST_LATENCY.labels("/predict").observe(latency)

        return PredictionResponse(
            transaction_id=None,
            is_fraud=is_fraud,
            fraud_probability=round(prob, 4),
            confidence=confidence,
        )

    except Exception as e:
        REQUEST_COUNT.labels("POST", "/predict", "500").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_metrics")
def update_metrics(recall: float, fpr: float, drift_score: float):
    """Called by monitoring system to update live model metrics."""
    MODEL_RECALL_GAUGE.set(recall)
    FALSE_POSITIVE_RATE_GAUGE.set(fpr)
    DATA_DRIFT_GAUGE.set(drift_score)
    return {"status": "updated", "recall": recall, "fpr": fpr, "drift": drift_score}


@app.get("/model_info")
def model_info():
    return {
        "model_name":  model_name,
        "version":     "1.0",
        "recall":      0.8125,
        "auc_roc":     0.9300,
        "threshold":   0.70,
        "status":      "production",
    }
