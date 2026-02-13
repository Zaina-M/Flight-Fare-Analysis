"""
Flight Price Prediction - REST API.

Simple Flask REST API for serving flight price predictions.
Loads the trained model and feature engineer, accepts JSON input,
and returns predicted fare.

Usage:
    python api.py                    # Run on default port 5000
    python api.py --port 8080        # Run on custom port

Endpoints:
    GET  /health          - Health check
    GET  /model/info      - Model metadata
    POST /predict          - Predict flight fare
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify
import joblib
import pickle

from config.config import MODELS_DIR, data_config

app = Flask(__name__)


# Model loading


_model = None
_feature_engineer = None
_model_meta = {}


def _load_artifacts():
    """Load model and feature engineer once at startup."""
    global _model, _feature_engineer, _model_meta

    # Find latest best model
    model_files = sorted(MODELS_DIR.glob("best_model_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not model_files:
        model_files = sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    if not model_files:
        raise FileNotFoundError("No trained model found in models/. Run `python main.py` first.")

    latest = model_files[-1]
    model_data = joblib.load(latest)
    _model = model_data.get("model") if isinstance(model_data, dict) else model_data
    _model_meta = {
        "model_file": latest.name,
        "model_type": type(_model).__name__,
        "loaded_at": datetime.now().isoformat(),
    }

    fe_path = MODELS_DIR / "feature_engineer.pkl"
    if fe_path.exists():
        with open(fe_path, "rb") as f:
            _feature_engineer = pickle.load(f)
        _model_meta["feature_engineer"] = True
        _model_meta["log_transform_target"] = getattr(_feature_engineer, "log_transform_target", False)
    else:
        _model_meta["feature_engineer"] = False



# Endpoints



@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API documentation."""
    return jsonify({
        "name": "Flight Price Prediction API",
        "version": "1.0",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /model/info": "Model metadata",
            "POST /predict": "Predict flight fare (JSON body required)"
        },
        "example_request": {
            "airline": "Biman Bangladesh",
            "source": "Dhaka",
            "destination": "Chittagong",
            "departure_date": "2025-03-15",
            "duration_hrs": 1.5,
            "days_before_departure": 14,
            "flight_class": "Economy",
            "booking_source": "Online Website",
            "seasonality": "Regular"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata."""
    return jsonify(_model_meta)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict flight fare from JSON payload.

    Expected JSON body:
    {
        "airline": "Biman Bangladesh",
        "source": "Dhaka",
        "destination": "Chittagong",
        "departure_date": "2025-03-15",
        "duration_hrs": 1.5,
        "days_before_departure": 14,
        "flight_class": "Economy",
        "booking_source": "Online Website",
        "seasonality": "Regular"
    }

    Returns:
    {
        "predicted_fare_bdt": 8500.50,
        "model": "GradientBoostingRegressor",
        "log_transformed": true
    }
    """
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    # Validate required fields
    required = [
        "airline", "source", "destination", "departure_date",
        "duration_hrs", "days_before_departure", "flight_class",
        "booking_source", "seasonality",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        # Build input DataFrame matching training schema
        input_df = pd.DataFrame({
            "Airline": [data["airline"]],
            "Source": [data["source"]],
            "Destination": [data["destination"]],
            "Departure Date & Time": [pd.Timestamp(data["departure_date"])],
            "Duration (hrs)": [float(data["duration_hrs"])],
            "Days Before Departure": [int(data["days_before_departure"])],
            "Class": [data["flight_class"]],
            "Booking Source": [data["booking_source"]],
            "Seasonality": [data["seasonality"]],
        })

        # Transform features
        if _feature_engineer is not None:
            input_transformed = _feature_engineer.transform(input_df, scale_features=True)
            input_transformed = input_transformed.fillna(0)
        else:
            return jsonify({"error": "Feature engineer not available"}), 503

        # Predict
        prediction = float(_model.predict(input_transformed)[0])

        # Reverse log transform if applicable
        log_flag = getattr(_feature_engineer, "log_transform_target", False)
        if log_flag:
            prediction = float(np.expm1(prediction))

        return jsonify({
            "predicted_fare_bdt": round(prediction, 2),
            "model": _model_meta.get("model_type", "unknown"),
            "log_transformed": log_flag,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Price Prediction REST API")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"Loading model artifacts from {MODELS_DIR}...")
    _load_artifacts()
    print(f"Model loaded: {_model_meta.get('model_type')} ({_model_meta.get('model_file')})")
    print(f"Starting API server on http://{args.host}:{args.port}")
    print(f"  POST /predict   - Predict flight fare")
    print(f"  GET  /health    - Health check")
    print(f"  GET  /model/info - Model info")

    app.run(host=args.host, port=args.port, debug=args.debug)
