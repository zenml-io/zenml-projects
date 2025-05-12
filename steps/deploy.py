# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import base64
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict

import modal
from zenml import log_metadata, step
from zenml.logger import get_logger

from constants import (
    APPROVED_NAME,
    DEPLOYMENT_INFO_NAME,
    EVALUATION_RESULTS_NAME,
    MODEL_PATH,
    PREPROCESS_PIPELINE_NAME,
)

logger = get_logger(__name__)

# Base image for deployment with necessary packages
image = modal.Image.debian_slim().pip_install(
    [
        "scikit-learn",
        "joblib",
        "pandas",
        "numpy",
        "requests",
        "whylogs",  # For data drift detection
    ]
)


@step(enable_cache=False)
def deploy_model(
    model_path: Annotated[str, MODEL_PATH],
    approved: Annotated[bool, APPROVED_NAME],
    evaluation_results: Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
    preprocess_pipeline: Annotated[Any, PREPROCESS_PIPELINE_NAME],
) -> Annotated[Dict[str, Any], DEPLOYMENT_INFO_NAME]:
    """Deploy model with monitoring and incident reporting (Articles 10, 17, 18).

    This step:
    1. Deploys the model to a Modal endpoint
    2. Sets up data drift monitoring (Article 17)
    3. Configures incident reporting webhook (Article 18)
    4. Logs complete deployment metadata for compliance documentation

    Args:
        model_path: Path to the trained model file
        approved: Whether deployment was approved by human oversight
        evaluation_results: Model evaluation metrics and fairness analysis
        preprocess_pipeline: The preprocessing pipeline used in training

    Returns:
        Dictionary with deployment information
    """
    if not approved:
        return {"status": "rejected", "reason": "Not approved by human oversight"}

    # Timestamp for versioning and record-keeping
    timestamp = datetime.now().isoformat()
    deployment_id = f"deployment_{timestamp.replace(':', '-')}"

    # Read model file and calculate checksum
    model_bytes = Path(model_path).read_bytes()
    model_checksum = hashlib.sha256(model_bytes).hexdigest()

    # Define Modal stub
    stub = modal.Stub(f"credit-scoring-{deployment_id}")

    # Helper functions for monitoring and incident reporting
    @stub.function(image=image)
    def report_incident(incident_data):
        """Report incidents to compliance team and log them (Article 18)."""
        import json
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

        log_metadata(metadata={"incident": incident})

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

    @stub.function(image=image, schedule=modal.Period(hours=24))
    def monitor_data_drift():
        """Daily monitoring for data drift (Article 17)."""
        import json
        from datetime import datetime

        import whylogs as why

        try:
            # In a real implementation, this would fetch recent predictions
            # and compare them to the reference profile

            # Simulate drift detection
            drift_detected = False  # This would be the result of actual comparison

            if drift_detected:
                # Report as incident if significant drift is detected
                report_incident(
                    {
                        "severity": "high",
                        "description": "Data drift detected",
                        "drift_metrics": {"distribution_shift": 0.25},  # Example metric
                    }
                )

                log_metadata(metadata={"drift_detected": drift_detected})

            return {"timestamp": datetime.now().isoformat(), "drift_detected": drift_detected}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Main prediction function
    @stub.function(image=image, cpu=2)
    def predict_proba(input_data):
        """Make predictions with the deployed model."""
        import base64
        from datetime import datetime

        import joblib
        import numpy as np
        import pandas as pd

        try:
            # Load model
            model = joblib.load(io.BytesIO(base64.b64decode(model_bytes_b64)))

            # Load preprocessing pipeline
            preprocess = joblib.load(io.BytesIO(base64.b64decode(preprocess_pipeline_b64)))

            # Convert input to DataFrame
            df = pd.DataFrame(input_data)

            # Preprocess input
            df_processed = preprocess.transform(df)

            # Get prediction
            probabilities = model.predict_proba(df_processed)[:, 1]

            # Log prediction for monitoring
            prediction_log = {
                "timestamp": datetime.now().isoformat(),
                "input_hash": hash(str(input_data)),
                "prediction_mean": float(np.mean(probabilities)),
            }

            log_metadata(metadata={"prediction_log": prediction_log})

            # In production, you'd store this log persistently

            return {
                "probabilities": probabilities.tolist(),
                "model_version": model_checksum[:8],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            # Log error for compliance
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "input_sample": str(input_data)[:100] + "..."
                if len(str(input_data)) > 100
                else str(input_data),
            }

            logger.error(f"Error in prediction: {error_log}")

            # In production, you'd store this error log persistently

            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Encode model and preprocessing pipeline for deployment
    model_bytes_b64 = base64.b64encode(model_bytes).decode()

    # Serialize preprocessing pipeline
    import joblib

    preprocess_bytes = joblib.dumps(preprocess_pipeline)
    preprocess_pipeline_b64 = base64.b64encode(preprocess_bytes).decode()

    # Deploy the Modal functions
    deployment = stub.deploy("prod")

    # Get service URLs
    prediction_url = f"{deployment.predict_proba.remote_url}"
    monitoring_url = f"{deployment.monitor_data_drift.remote_url}"
    incident_url = f"{deployment.report_incident.remote_url}"

    # Create deployment record for compliance
    deployment_record = {
        "deployment_id": deployment_id,
        "timestamp": timestamp,
        "model_checksum": model_checksum,
        "model_path": model_path,
        "performance_metrics": evaluation_results.get("metrics", {}),
        "endpoints": {
            "prediction": prediction_url,
            "monitoring": monitoring_url,
            "incident_reporting": incident_url,
        },
        "monitoring_schedule": "daily",
        "stub_name": stub.name,
    }

    # Save deployment record
    deployment_dir = Path("compliance/deployment_records")
    deployment_dir.mkdir(parents=True, exist_ok=True)

    with open(deployment_dir / f"{deployment_id}.json", "w") as f:
        json.dump(deployment_record, f, indent=2)

    # Create model card (for Article 13 transparency)
    model_card = {
        "model_id": model_checksum[:8],
        "name": "Credit Risk Assessment Model",
        "version": timestamp,
        "description": "This model assesses credit risk for loan applications",
        "date_created": timestamp,
        "performance_metrics": {
            "accuracy": evaluation_results.get("metrics", {}).get("accuracy"),
            "auc": evaluation_results.get("metrics", {}).get("auc"),
            "f1": evaluation_results.get("metrics", {}).get("f1"),
        },
        "intended_use": "Credit scoring for loan application evaluation",
        "limitations": "This model should be used as a decision support tool only and not as the sole basis for credit decisions",
        "fairness_considerations": "Model has been evaluated for bias across protected attributes",
        "contact_information": "compliance@example.com",
    }

    # Save model card
    with open(deployment_dir / f"model_card_{deployment_id}.json", "w") as f:
        json.dump(model_card, f, indent=2)

    # Log metadata for compliance documentation
    log_metadata(
        metadata={
            f"deployment_{timestamp.replace(':', '-')}": deployment_record,
            f"model_card_{timestamp.replace(':', '-')}": model_card,
        }
    )

    print("✅ Model deployed successfully!")
    print(f"📊 Prediction endpoint: {prediction_url}")
    print(f"🔍 Monitoring active: {monitoring_url}")
    print(f"🚨 Incident reporting: {incident_url}")

    return deployment_record
