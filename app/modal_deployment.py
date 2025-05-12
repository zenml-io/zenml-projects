# app/modal_deployment.py
#!/usr/bin/env python3
"""EU AI Act compliant credit scoring model deployment with Modal."""

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

os.environ["MODAL_AUTOMOUNT"] = "false"

import modal
from fastapi import FastAPI, HTTPException

from app.schemas import (
    ApiInfo,
    CreditScoringFeatures,
    IncidentReport,
    IncidentResponse,
    MonitorResponse,
    PredictionResponse,
)

# -- Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit-scoring-deployer")

# -- Configuration ─────────────────────────────────────────────────────────────
APP_NAME = "credit-score-predictor"
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models/model.pkl")
PREPROCESS_PATH = os.getenv("PREPROCESS_PATH", "/mnt/pipelines/preprocess_pipeline.pkl")
VOLUME_NAME = os.getenv("VOLUME_NAME", "credit-scoring")
SECRET_NAME = os.getenv("SECRET_NAME", "credit-scoring-secrets")
MODAL_ENVIRONMENT = os.getenv("MODAL_ENVIRONMENT", "main")


# -- App & Image ─────────────────────────────────────────────────────────────
def create_modal_app(python_version: str = "3.12.9"):
    """Create a Modal app for a given framework, stage, and volume name."""
    # Create base image
    base_image = (
        modal.Image.debian_slim(python_version=python_version)
        .pip_install(
            "scikit-learn",
            "joblib",
            "pandas",
            "numpy",
            "requests",
            "whylogs",
            "fastapi[standard]",
            "uvicorn",
        )
        .add_local_python_source("app")
    )

    app_config = {
        "image": base_image,
        "secrets": [modal.Secret.from_name(SECRET_NAME)],
    }

    try:
        volume = modal.Volume.from_name(VOLUME_NAME)
        app_config["volumes"] = {"/mnt": volume}
        logger.info(f"Added volume {VOLUME_NAME} to app")
    except Exception as e:
        logger.warning(f"Could not add volume {VOLUME_NAME}: {e}")

    return modal.App(APP_NAME, **app_config)


# create app
app = create_modal_app()


# ─── Functions ────────────────────────────────────────────────────────────────
def _load_model() -> Any:
    """Load model into memory from Modal volume."""
    import pickle

    model_path = MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join("/mnt", model_path)

        if os.path.exists(model_path):
            logger.info(f"Loading sklearn model from volume: {model_path}")
            with open(model_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")


def _load_pipeline() -> Any:
    """Load preprocessing pipeline into memory from Modal volume."""
    import pickle

    pipeline_path = PREPROCESS_PATH
    if not os.path.isabs(pipeline_path):
        pipeline_path = os.path.join("/mnt", pipeline_path)

    if os.path.exists(pipeline_path):
        logger.info(f"Loading preprocessing pipeline from volume: {pipeline_path}")
        with open(pipeline_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")


def _report_incident(incident_data: dict, model_checksum: str) -> dict:
    """Report incidents to compliance team and log them (Article 18)."""
    from datetime import datetime

    import requests

    # Format incident report
    incident = {
        "incident_id": f"incident_{datetime.now().isoformat().replace(':', '-')}",
        "timestamp": datetime.now().isoformat(),
        "model_version": model_checksum[:8],
        "severity": incident_data.get("severity", "medium"),
        "description": incident_data.get("description", "Unspecified incident"),
        "data": incident_data,
    }

    # Save incident to log
    try:
        # Log to persistent storage
        with open("/tmp/incident_log.json", "a") as f:
            f.write(json.dumps(incident) + "\n")

        # Optional: Send to webhook
        webhook_url = incident_data.get("webhook_url")
        if webhook_url:
            requests.post(webhook_url, json=incident)

        return {
            "status": "reported",
            "incident_id": incident["incident_id"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _monitor_data_drift() -> dict:
    """Daily monitoring for data drift (Article 17)."""
    from datetime import datetime

    import whylogs as why

    try:
        # In a real implementation, this would fetch recent predictions
        # and compare them to the reference profile

        # Simulate drift detection
        drift_detected = False  # This would be the result of actual comparison

        if drift_detected:
            report_incident(
                {
                    "severity": "high",
                    "description": "Data drift detected",
                }
            )

        return {"timestamp": datetime.now().isoformat(), "drift_detected": drift_detected}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _predict(
    input_data: list,
    model_checksum: str = None,
) -> dict:
    """Make predictions with the deployed model."""
    from datetime import datetime

    import numpy as np
    import pandas as pd

    try:
        model = load_model.remote()
        pipeline = load_pipeline.remote()

        if model is None:
            return {"error": "Model could not be loaded", "timestamp": datetime.now().isoformat()}

        if pipeline is None:
            return {
                "error": "Preprocessing pipeline could not be loaded",
                "timestamp": datetime.now().isoformat(),
            }

        # Process data and make prediction
        # Convert the input dict to DataFrame - removing the 'scale__' prefix if needed
        input_dict = {}
        for key, value in input_data.items():
            # Handle both with and without scale__ prefix for flexibility
            clean_key = key.replace("scale__", "") if key.startswith("scale__") else key
            input_dict[clean_key] = value

        df = pd.DataFrame([input_dict])

        # Preprocess input
        X = pipeline.transform(df)

        # Get prediction
        probs = model.predict_proba(X)[:, 1]

        # Log prediction
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": hash(str(input_data)),
            "prediction_mean": float(np.mean(probs)),
        }
        logger.info(f"Prediction log: {prediction_log}")

        # Assess risk score based on probability
        risk_assessment = {
            "risk_score": float(probs[0]),
            "risk_level": "high" if probs[0] > 0.7 else "medium" if probs[0] > 0.3 else "low",
        }

        return {
            "probabilities": probs.tolist(),
            "model_version": model_checksum[:8] if model_checksum else "unknown",
            "timestamp": datetime.now().isoformat(),
            "risk_assessment": risk_assessment,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


def _create_fastapi_app() -> FastAPI:
    """Create FastAPI app for Credit Score Predictions."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # warm the model when the container starts
        try:
            load_model.remote()
            load_pipeline.remote()
            logger.info("Model and pipeline loaded on startup")
        except Exception as e:
            logger.error(f"Warm-up failed: {e}")
        yield

    web_app = FastAPI(
        title="Credit Scoring API",
        version="1.0.0",
        description="EU AI Act compliant Credit Scoring API",
        lifespan=lifespan,
    )

    @web_app.get("/", response_model=ApiInfo)
    async def root() -> Dict:
        """Root endpoint with API info."""
        logger.info("Root endpoint called")
        actual_url = str(fastapi_app.web_url).rstrip("/")

        return {
            "message": "Credit Scoring API - EU AI Act Compliant",
            "endpoints": {
                "root": actual_url,
                "health": f"{actual_url}/health",
                "predict": f"{actual_url}/predict",
                "monitor": f"{actual_url}/monitor",
                "incident": f"{actual_url}/incident",
            },
            "app_name": APP_NAME,
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint."""
        logger.info("Health check called")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    @web_app.post("/incident", response_model=IncidentResponse)
    async def report_incident_endpoint(incident_data: IncidentReport) -> Dict:
        """Report an incident to the compliance system."""
        logger.info(
            f"Incident report received: {incident_data.severity} - {incident_data.description}"
        )
        try:
            model_info = getattr(load_model.remote(), "_model_info", {})
            model_checksum = model_info.get("checksum", "unknown")

            result = report_incident.remote(incident_data.dict(), model_checksum)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Add timestamp if not present
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()

            return result
        except Exception as e:
            logger.exception(f"Incident reporting error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Incident reporting failed: {str(e)}")

    @web_app.get("/monitor", response_model=MonitorResponse)
    async def monitor_endpoint() -> Dict:
        """Monitor data drift and model performance."""
        logger.info("Monitor endpoint called - running drift detection")
        try:
            result = monitor_data_drift.remote()
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        except Exception as e:
            logger.exception(f"Monitoring error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(features: CreditScoringFeatures) -> Dict:
        """Make a credit scoring prediction."""
        logger.info("Prediction request received")
        try:
            # Convert Pydantic model to dict for prediction
            input_data = features.dict()

            # Get model info for checksum
            model_info = getattr(load_model.remote(), "_model_info", {})
            model_checksum = model_info.get("checksum", "unknown")

            # Call prediction function
            result = predict.remote(input_data, model_checksum)

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        except Exception as e:
            logger.exception(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return web_app


# Create initial functions with the initial app
load_model = app.function()(_load_model)
load_pipeline = app.function()(_load_pipeline)
predict = app.function()(_predict)
report_incident = app.function()(_report_incident)
monitor_data_drift = app.function()(_monitor_data_drift)
fastapi_app = app.function()(modal.asgi_app(label=APP_NAME)(_create_fastapi_app))


@app.local_entrypoint()
def main(
    model_path: str,
    evaluation_results: Dict,
    preprocess_pipeline: Any,
):
    """Deploy a model to Modal; returns (deployment_record, model_card)."""
    ts = datetime.now().isoformat()
    deployment_id = f"deployment_{ts.replace(':', '-')}"

    # Generate checksum
    model_bytes = Path(model_path).read_bytes()
    model_checksum = hashlib.sha256(model_bytes).hexdigest()

    global app, load_model, load_pipeline, predict, fastapi_app, report_incident, monitor_data_drift
    # Create a fresh app instance

    app = create_modal_app()
    load_model = app.function()(_load_model)
    load_pipeline = app.function()(_load_pipeline)
    predict = app.function()(_predict)
    report_incident = app.function()(_report_incident)
    monitor_data_drift = app.function()(_monitor_data_drift)
    fastapi_app = app.function()(modal.asgi_app(label=APP_NAME)(_create_fastapi_app))

    # Deploy (invoke Modal)
    from modal.output import enable_output
    from modal.runner import deploy_app

    with enable_output():
        deploy_result = deploy_app(
            app,
            name=APP_NAME,
            environment_name=MODAL_ENVIRONMENT,
        )

    # Get the URL of the deployed app
    fastapi_url = None
    # Method 1: Try to get URL directly from the fastapi_app
    try:
        # Access the web_url attribute directly after deployment
        fastapi_url = fastapi_app.web_url
        logger.info(f"Got URL from fastapi_app.web_url: {fastapi_url}")
    except Exception as e:
        logger.warning(f"Could not get URL from fastapi_app directly: {e}")

        # Method 2: Construct URL from workspace name and label
        workspace_name = getattr(deploy_result, "workspace_name", "marwan-ext")
        fastapi_url = f"https://{workspace_name}-{MODAL_ENVIRONMENT}--{APP_NAME}.modal.run"
        logger.info(f"Constructed URL with stage: {fastapi_url}")

    logger.info(f"Serving Credit Scoring Model on {APP_NAME}")
    logger.info(f"Deploy App Result: {deploy_result}")
    logger.info(f"API URL: {fastapi_url}")

    urls = {
        "root": fastapi_url,
        "predict": f"{fastapi_url}/predict",
        "monitor": f"{fastapi_url}/monitor",
        "incident": f"{fastapi_url}/incident",
        "health": f"{fastapi_url}/health",
    }

    # Record deployment metadata
    deployment_record = {
        "deployment_id": deployment_id,
        "timestamp": ts,
        "model_checksum": model_checksum,
        "model_path": model_path,
        "endpoints": urls,
        "metrics": evaluation_results.get("metrics", {}),
        "app_name": APP_NAME,
    }

    # Add preprocessing pipeline info if available
    if preprocess_pipeline is not None:
        import pickle

        # Generate checksum for preprocessing pipeline
        preprocess_bytes = pickle.dumps(preprocess_pipeline)
        preprocess_checksum = hashlib.sha256(preprocess_bytes).hexdigest()

        deployment_record["preprocess_pipeline_checksum"] = preprocess_checksum

    # Create model card
    model_card = {
        "model_id": model_checksum[:8],
        "name": "Credit Risk Assessment Model",
        "version": ts,
        "description": "This model assesses credit risk for loan applications",
        "date_created": ts,
        "performance_metrics": {},
    }

    # Add metrics if available
    if evaluation_results and "metrics" in evaluation_results:
        model_card["performance_metrics"] = {
            "accuracy": evaluation_results.get("metrics", {}).get("accuracy"),
            "auc": evaluation_results.get("metrics", {}).get("auc"),
            "f1": evaluation_results.get("metrics", {}).get("f1"),
        }

    # Add fairness metrics if available
    if evaluation_results and "fairness" in evaluation_results:
        model_card["fairness_metrics"] = evaluation_results["fairness"]

    # Add standard fields
    model_card.update(
        {
            "intended_use": "Credit scoring for loan application evaluation",
            "limitations": "This model should be used as a decision support tool only and not as the sole basis for credit decisions",
            "fairness_considerations": "Model has been evaluated for bias across protected attributes",
            "contact_information": "compliance@example.com",
        }
    )

    logger.info("✅ Model deployed successfully with endpoints: %s", deployment_record["endpoints"])
    return deployment_record, model_card


def run_deployment_entrypoint(**kwargs) -> Any:
    """Wrapper to invoke local_entrypoint for external callers."""
    return main(**kwargs)
