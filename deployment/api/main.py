from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
from pathlib import Path

from transformers import CrimeFeatureEngineer

# ============================================================
# LOGGING SETUP
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Separate logger for predictions (structured for analysis)
prediction_logger = logging.getLogger("predictions")
prediction_handler = logging.FileHandler(LOG_DIR / "predictions.jsonl")
prediction_handler.setFormatter(logging.Formatter('%(message)s'))
prediction_logger.addHandler(prediction_handler)
prediction_logger.setLevel(logging.INFO)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Crime Hotspot Forecast API",
    description="Crime hotspot forecast using Logistic Regression, Random Forest and XGBoost models",
    version="1.0.0"
)

# ============================================================
# LOAD PIPELINES & METADATA
# ============================================================
MODEL_DIR = Path("models")

try:
    fe = joblib.load(os.path.join(MODEL_DIR, 'feature_engineer.pkl'))
    
    lr_pipeline = joblib.load(MODEL_DIR / "lr.pkl")
    rf_pipeline = joblib.load(MODEL_DIR / "rf.pkl")
    xgb_pipeline = joblib.load(MODEL_DIR / "xgb.pkl")
    
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    
    with open(MODEL_DIR / "feature_stats.json") as f:
        feature_stats = json.load(f)
    
    # Extract feature info from metadata 
    FEATURE_COLS = metadata.get("feature_columns", metadata.get("features", []))
    CATEGORICAL_COLS = metadata.get("categorical_columns", ["primary_type"])
    PRIMARY_TYPE_CLASSES = metadata.get("primary_type_classes", [])
    
    logger.info("Pipelines loaded successfully!")
    logger.info(f"   Feature columns: {FEATURE_COLS}")
    logger.info(f"   Primary type classes: {PRIMARY_TYPE_CLASSES}")
    
except Exception as e:
    logger.error(f"Failed to load pipelines: {e}")
    raise

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class CrimePredictionInput(BaseModel):
    date: datetime = Field(default_factory=datetime.now)                         
    primary_type: str                          
    latitude: float              
    longitude: float              
    d1_count: int = 15                      # Previous day crime count
    d7_count: int = 15                      # Past 7-day crime count
    d7_avg: float = 14.50                   # Past 7-day crime count avg
    d7_std: float = 3.84                    # Past 7-day crime count std dev
    arrest_count: float = 14.26             # Past 7-day arrest count
    domestic_count: float = 17.96           # Past 7-day domestic-related count
    d30_avg: float = 14.55                  # Past 30-day crime count avg
    d30_std: float = 4.09                   # Past 30-day crime count std dev
    
    class Config:
        json_schema_extra = {
            "example": {
                "date": "2026-04-08T12:00:00",
                "primary_type": "THEFT",
                "latitude": 41.8781,
                "longitude": -87.6298,
                "d1_count": 15,
                "d1_count": 15,
                "d7_avg": 14.50,
                "d7_std": 3.84,
                "arrest_count": 14.26,
                "domestic_count": 17.96,
                "d30_avg": 14.55,
                "d30_std": 4.09
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response with individual and ensemble results."""
    transaction_id: str
    timestamp: str
    
    # Individual model predictions
    logistic_regression_prediction: int
    logistic_regression_probability: float
    random_forest_prediction: int
    random_forest_probability: float
    xgboost_prediction: int
    xgboost_probability: float
    
    # Ensemble prediction
    ensemble_prediction: int
    ensemble_probability: float
    ensemble_verdict: str
    
# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log_prediction(request_id: str, input_data: dict, prediction: dict):
    """Log prediction for monitoring and analysis."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "input": input_data,
        "predictions": {
            "logistic_regression": {"pred": prediction["lr_pred"], "prob": prediction["lr_prob"]},
            "random_forest": {"pred": prediction["rf_pred"], "prob": prediction["rf_prob"]},
            "xgboost": {"pred": prediction["xgb_pred"], "prob": prediction["xgb_prob"]},
            "ensemble": {"pred": prediction["ensemble_pred"], "prob": prediction["ensemble_prob"]}
        }
    }
    
    # Write to JSONL file (one JSON object per line for easy parsing)
    prediction_logger.info(json.dumps(log_entry))


# Global request counter
request_counter = 0

# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
def root():
    """API health check."""
    return {
        "status": "healthy",
        "message": "Crime Hotspot Forecast API is running",
        "version": "1.0.0",
        "models": ["Logistic Regression Pipeline", "Random Forest Pipeline", "XGBoost Pipeline"],
        "ensemble_method": "Soft Voting (Average)"
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "pipelines_loaded": True,
        "logistic_regression_pipeline": "ready",
        "random_forest_pipeline": "ready",
        "xgboost_pipeline": "ready",
        "total_predictions": request_counter
    }


@app.get("/model-info")
def model_info():
    """Get model metadata and metrics."""
    return {
        "feature_columns": FEATURE_COLS,
        "categorical_columns": CATEGORICAL_COLS,
        "primary_type_classes": PRIMARY_TYPE_CLASSES,
        "metrics": metadata.get("metrics", {}),
        "training_samples": metadata.get("training_samples", 0),
        "test_samples": metadata.get("test_samples", 0),
        # "pipeline_steps": metadata.get("pipeline_steps", [])
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CrimePredictionInput):
    """
    Predict crime risk for a spatiotemporal input.
    
    Returns predictions from:
    - Logistic Regression Pipeline 
    - Random Forest Pipeline
    - XGBoost Pipeline 
    - Ensemble (soft voting average)
    """
    global request_counter
    request_counter += 1
    request_id = f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_counter:06d}"
    
    try:
        # Validate primary_type
        if payload.primary_type not in PRIMARY_TYPE_CLASSES:
            logger.warning(f"[{request_id}] Unknown crime reporting type: {payload.primary_type}")
        
        input_data = {
            'date': [payload.date],
            'primary_type': [payload.primary_type],
            'latitude': [payload.latitude],
            'longitude': [payload.longitude],
            'd1_count': [payload.d1_count],
            'd7_count': [payload.d7_count],
            'd7_avg': [payload.d7_avg],
            'd7_std': [payload.d7_std],
            'arrest_count': [payload.arrest_count],
            'domestic_count': [payload.domestic_count],
            'd30_avg': [payload.d30_avg],
            'd30_std': [payload.d30_std],
        }
        
        # Create DataFrame with correct column order
        X = pd.DataFrame(input_data)[FEATURE_COLS]
        
        # Create features from payload
        X = fe.transform(X)

        # Get predictions 
        lr_prob = float(lr_pipeline.predict_proba(X)[0, 1])
        lr_pred = int(lr_prob >= 0.5)
        
        rf_prob = float(rf_pipeline.predict_proba(X)[0, 1])
        rf_pred = int(rf_prob >= 0.5)

        xgb_prob = float(xgb_pipeline.predict_proba(X)[0, 1])
        xgb_pred = int(xgb_prob >= 0.5)
        
        # Ensemble (soft voting)
        ensemble_prob = (lr_prob + rf_prob + xgb_prob) / 3
        ensemble_pred = int(ensemble_prob >= 0.5)
        
        # Determine verdict
        if ensemble_prob >= 0.5:
            verdict = "HIGH CRIME"
        else:
            verdict = "LOW CRIME"
        
        # Log prediction
        log_prediction(
            request_id=request_id,
            input_data=jsonable_encoder(payload.model_dump()),
            prediction={
                "lr_pred": lr_pred, "lr_prob": lr_prob,
                "rf_pred": rf_pred, "rf_prob": rf_prob,
                "xgb_pred": xgb_pred, "xgb_prob": xgb_prob,
                "ensemble_pred": ensemble_pred, "ensemble_prob": ensemble_prob
            }
        )
        
        logger.info(f"[{request_id}] Prediction: {verdict} (prob={ensemble_prob:.3f})")
        
        return PredictionResponse(
            transaction_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            logistic_regression_prediction=lr_pred,
            logistic_regression_probability=round(lr_prob, 4),
            random_forest_prediction=rf_pred,
            random_forest_probability=round(rf_prob, 4),
            xgboost_prediction=xgb_pred,
            xgboost_probability=round(xgb_prob, 4),
            ensemble_prediction=ensemble_pred,
            ensemble_probability=round(ensemble_prob, 4),
            ensemble_verdict=verdict
            # drift_warnings=drift_warnings
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary")
def get_log_summary():
    """Get summary of logged predictions for monitoring."""
    log_file = LOG_DIR / "predictions.jsonl"
    
    if not log_file.exists():
        return {"message": "No predictions logged yet", "total": 0}
    
    total = 0
    high_crime_count = 0
    # drift_count = 0
    
    with open(log_file) as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                if entry["predictions"]["ensemble"]["pred"] == 1:
                    high_crime_count += 1
            except:
                continue
    
    return {
        "total_predictions": total,
        "high_crime_predictions": high_crime_count,
        "high_crime_rate": round(high_crime_count / total * 100, 2) if total > 0 else 0,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
