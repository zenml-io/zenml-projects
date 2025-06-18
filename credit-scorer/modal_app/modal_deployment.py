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

import requests
from src.constants.config import SlackConfig as SC

os.environ["MODAL_AUTOMOUNT"] = "false"

import modal
from fastapi import FastAPI, HTTPException
from src.constants.config import ModalConfig

from modal_app.schemas import (
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
volume_metadata = ModalConfig.get_volume_metadata()
APP_NAME = os.getenv("APP_NAME", volume_metadata["app_name"])
VOLUME_NAME = os.getenv("VOLUME_NAME", volume_metadata["volume_name"])
SECRET_NAME = os.getenv("SECRET_NAME", volume_metadata["secret_name"])
ENVIRONMENT = os.getenv(
    "MODAL_ENVIRONMENT", volume_metadata["environment_name"]
)
# Paths within the container (prefixed with /mnt)
MODEL_PATH = os.getenv("MODEL_PATH", f"/mnt/{volume_metadata['model_path']}")
PREPROCESS_PATH = os.getenv(
    "PREPROCESS_PATH", f"/mnt/{volume_metadata['preprocess_pipeline_path']}"
)


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
            "lightgbm",
            "requests",
            "whylogs",
            "fastapi[standard]",
            "uvicorn",
        )
        .add_local_python_source("modal_app")
        .add_local_file(
            "src/constants/config.py",
            remote_path="/root/src/constants/config.py",
        )
        .add_local_file(
            "src/constants/annotations.py",
            remote_path="/root/src/constants/annotations.py",
        )
        .add_local_file(
            "src/utils/storage.py",
            remote_path="/root/src/utils/storage.py",
        )
    )

    app_config = {
        "image": base_image,
    }

    # Only add secrets if Slack notifications are explicitly enabled
    enable_slack = os.getenv("ENABLE_SLACK", "false").lower() == "true"
    if enable_slack:
        try:
            app_config["secrets"] = [modal.Secret.from_name(SECRET_NAME)]
            logger.info(f"Added secret {SECRET_NAME} to Modal app")
        except Exception as e:
            logger.warning(f"Could not add secret {SECRET_NAME}: {e}")
            logger.info(
                "Continuing without secrets - Slack notifications will be disabled"
            )
    else:
        logger.info(
            "Slack notifications disabled by default - Modal app created without secrets"
        )

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
    import joblib

    pipeline_path = PREPROCESS_PATH
    if not os.path.isabs(pipeline_path):
        pipeline_path = os.path.join("/mnt", pipeline_path)

    if os.path.exists(pipeline_path):
        logger.info(
            f"Loading preprocessing pipeline from volume: {pipeline_path}"
        )
        return joblib.load(pipeline_path)
    else:
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")


def _report_incident(incident_data: dict, model_checksum: str) -> dict:
    """Report incidents to compliance team and log them (Article 18)."""
    # Format incident report
    incident = {
        "incident_id": f"incident_{datetime.now().isoformat().replace(':', '-')}",
        "timestamp": datetime.now().isoformat(),
        "model_name": "credit_scoring_model",
        "model_version": model_checksum,
        "severity": incident_data.get("severity", "medium"),
        "description": incident_data.get(
            "description", "Unspecified incident"
        ),
        "source": "modal_api",
        "data": incident_data,
    }

    try:
        # 1. Append to local log (if accessible)
        incident_log_path = "docs/risk/incident_log.json"
        try:
            existing = []
            if Path(incident_log_path).exists():
                try:
                    with open(incident_log_path, "r") as f:
                        existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []

            existing.append(incident)
            with open(incident_log_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write to local incident log: {e}")

        # 2. Direct Slack notification for high/critical severity (not using ZenML)
        enable_slack = os.getenv("ENABLE_SLACK", "false").lower() == "true"
        if incident["severity"] in ("high", "critical") and enable_slack:
            try:
                slack_token = os.getenv("SLACK_BOT_TOKEN")
                slack_channel = os.getenv("SLACK_CHANNEL_ID", SC.CHANNEL_ID)

                if slack_token and slack_channel:
                    emoji = {"high": "🔴", "critical": "🚨"}[
                        incident["severity"]
                    ]
                    message = (
                        f"{emoji} *Incident from Modal API:* {incident['description']}\n"
                        f">*Severity:* {incident['severity']}\n"
                        f">*Source:* {incident['source']}\n"
                        f">*Model Version:* {incident['model_version']}\n"
                        f">*Time:* {incident['timestamp']}\n"
                        f">*ID:* {incident['incident_id']}"
                    )

                    # Direct Slack API call
                    response = requests.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {slack_token}"},
                        json={
                            "channel": slack_channel,
                            "text": message,
                            "username": "Modal Incident Bot",
                        },
                    )

                    if response.status_code == 200:
                        incident["slack_notified"] = True
                        logger.info("Slack notification sent successfully")
                    else:
                        logger.warning(
                            f"Slack notification failed: {response.text}"
                        )
                else:
                    logger.info(
                        "Slack credentials not available, skipping notification"
                    )
            except Exception as e:
                logger.warning(f"Failed to send Slack notification: {e}")
        elif not enable_slack:
            logger.info(
                "Slack notifications disabled (use --enable-slack flag to enable)"
            )

        return {
            "status": "reported",
            "incident_id": incident["incident_id"],
            "slack_notified": incident.get("slack_notified", False),
        }

    except Exception as e:
        logger.error(f"Error reporting incident: {e}")
        return {"status": "error", "message": str(e)}


def _monitor_data_drift() -> dict:
    """Daily monitoring for data drift (Article 17)."""
    from datetime import datetime

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

        return {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
        }
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
            return {
                "error": "Model could not be loaded",
                "timestamp": datetime.now().isoformat(),
            }

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
            clean_key = (
                key.replace("scale__", "")
                if key.startswith("scale__")
                else key
            )
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
            "risk_level": "high"
            if probs[0] > 0.7
            else "medium"
            if probs[0] > 0.3
            else "low",
        }

        return {
            "probabilities": probs.tolist(),
            "model_version": model_checksum[:8]
            if model_checksum
            else "unknown",
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
    async def root() -> ApiInfo:
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
            "model": APP_NAME,
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
    async def report_incident_endpoint(
        incident_data: IncidentReport,
    ) -> IncidentResponse:
        """Report an incident to the compliance system."""
        logger.info(
            f"Incident report received: {incident_data.severity} - {incident_data.description}"
        )
        try:
            model_info = getattr(load_model.remote(), "_model_info", {})
            model_checksum = model_info.get("checksum", "unknown")

            result = report_incident.remote(
                incident_data.model_dump(), model_checksum
            )
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Add timestamp if not present
            if "timestamp" not in result:
                result["timestamp"] = datetime.now().isoformat()

            return result
        except Exception as e:
            logger.exception(f"Incident reporting error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Incident reporting failed: {str(e)}"
            )

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
            raise HTTPException(
                status_code=500, detail=f"Monitoring failed: {str(e)}"
            )

    @web_app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(features: CreditScoringFeatures) -> Dict:
        """Make a credit scoring prediction."""
        logger.info("Prediction request received")
        try:
            # Convert Pydantic model to dict for prediction
            input_data = features.model_dump()

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
            raise HTTPException(
                status_code=500, detail=f"Prediction failed: {str(e)}"
            )

    return web_app


# Create initial functions with the initial app
load_model = app.function()(_load_model)
load_pipeline = app.function()(_load_pipeline)
predict = app.function()(_predict)
report_incident = app.function()(_report_incident)
monitor_data_drift = app.function()(_monitor_data_drift)
fastapi_app = app.function()(
    modal.asgi_app(label=APP_NAME)(_create_fastapi_app)
)


@app.local_entrypoint()
def main(
    model: Any,
    evaluation_results: Dict,
    preprocess_pipeline: Any,
):
    """Deploy a model to Modal; returns (deployment_record, model_card)."""
    ts = datetime.now().isoformat()
    deployment_id = f"deployment_{ts.replace(':', '-')}"

    # Generate model checksum without saving again
    import pickle

    model_bytes = pickle.dumps(model)
    model_checksum = hashlib.sha256(model_bytes).hexdigest()

    global app, load_model, load_pipeline, predict, fastapi_app, report_incident, monitor_data_drift
    # Create a fresh app instance

    app = create_modal_app()
    load_model = app.function()(_load_model)
    load_pipeline = app.function()(_load_pipeline)
    predict = app.function()(_predict)
    report_incident = app.function()(_report_incident)
    monitor_data_drift = app.function()(_monitor_data_drift)
    fastapi_app = app.function()(
        modal.asgi_app(label=APP_NAME)(_create_fastapi_app)
    )

    # Deploy (invoke Modal)
    from modal.output import enable_output
    from modal.runner import deploy_app

    with enable_output():
        deploy_result = deploy_app(
            app,
            name=APP_NAME,
            environment_name=ENVIRONMENT,
        )

    # Get the URL of the deployed app
    fastapi_url = None
    # Method 1: Try to get URL directly from the fastapi_app
    try:
        # Access the web_url attribute directly after deployment
        fastapi_url = fastapi_app.get_web_url()
        logger.info(f"Got URL from fastapi_app.get_web_url: {fastapi_url}")
    except Exception as e:
        logger.warning(f"Could not get URL from fastapi_app directly: {e}")

    # Method 2: Construct URL from workspace name and label
    workspace_name = getattr(deploy_result, "workspace_name", "marwan-ext")

    if ENVIRONMENT == "main":
        # Default environment doesn't include the suffix
        fastapi_url = f"https://{workspace_name}--{APP_NAME}.modal.run"
    else:
        # Non-default environments include the suffix
        fastapi_url = (
            f"https://{workspace_name}-{ENVIRONMENT}--{APP_NAME}.modal.run"
        )

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
        "model_path": MODEL_PATH,
        "endpoints": urls,
        "metrics": evaluation_results.get("metrics", {}),
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
    }

    # Add preprocessing pipeline info if available
    if preprocess_pipeline is not None:
        import pickle

        # Generate checksum for preprocessing pipeline
        preprocess_bytes = pickle.dumps(preprocess_pipeline)
        preprocess_checksum = hashlib.sha256(preprocess_bytes).hexdigest()

        # Add to deployment record
        deployment_record["preprocess_pipeline_checksum"] = preprocess_checksum
        deployment_record["preprocess_pipeline_path"] = PREPROCESS_PATH

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

    logger.info(
        "✅ Model deployed successfully with endpoints: %s",
        deployment_record["endpoints"],
    )
    return deployment_record, model_card


def run_deployment_entrypoint(**kwargs) -> Any:
    """Wrapper to invoke local_entrypoint for external callers.

    Note: This function handles saving the model and preprocessing pipeline
    to Modal volume, no other artifacts need to be saved separately.
    """
    return main(**kwargs)
