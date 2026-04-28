import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from pathlib import Path

REGISTRY_MODEL_NAME = "hotel-booking-classifier"
LOCAL_MODEL_PATH = Path("models/random_forest.pkl")

_pipeline = None


def load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # @champion = alias assigned in MLflow UI to the best production model
        # Use @alias format (new) not /Stage format (deprecated in MLflow 2.9+)
        model_uri = f"models:/{REGISTRY_MODEL_NAME}@champion"
        _pipeline = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from MLflow Registry: {model_uri}")
    except Exception as e:
        # Fallback to local file if registry is unreachable
        print(f"Registry unavailable ({e}), loading from local file.")
        _pipeline = joblib.load(LOCAL_MODEL_PATH)
        print(f"Model loaded from local path: {LOCAL_MODEL_PATH}")

    return _pipeline


def engineer_features(data: dict) -> dict:
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]
    data["total_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    return data


def predict(booking: dict) -> dict:
    pipeline = load_pipeline()

    booking = engineer_features(booking.copy())
    df = pd.DataFrame([booking])

    probability = pipeline.predict_proba(df)[0][1]
    will_cancel = bool(probability >= 0.5)

    if probability >= 0.7:
        confidence = "High"
    elif probability >= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "will_cancel": will_cancel,
        "cancellation_probability": round(float(probability), 4),
        "confidence": confidence
    }
