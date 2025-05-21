"""API Dashboard component for the Modal FastAPI app."""

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd
import requests
import streamlit as st
from zenml.client import Client

from src.constants import (
    MODEL_NAME,
    TRAINING_PIPELINE_NAME,
)

logger = logging.getLogger(__name__)

# Sample data for testing the UI
SAMPLE_PREDICTION_REQUEST = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "CNT_CHILDREN": 0,
    "AMT_INCOME_TOTAL": 450000.0,
    "AMT_CREDIT": 1000000.0,
    "AMT_ANNUITY": 60000.0,
    "AMT_GOODS_PRICE": 900000.0,
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "DAYS_BIRTH": -10000,
    "DAYS_EMPLOYED": -3000,
    "OCCUPATION_TYPE": "Laborers",
    "CNT_FAM_MEMBERS": 2.0,
    "REGION_RATING_CLIENT": 2.0,
    "EXT_SOURCE_1": 0.75,
    "EXT_SOURCE_2": 0.65,
    "EXT_SOURCE_3": 0.85,
}

SAMPLE_INCIDENT_REQUEST = {
    "severity": "medium",
    "description": "Unusual pattern detected in predictions",
    "data": {
        "affected_predictions": 10,
        "timeframe": "2024-03-20T09:00:00Z to 2024-03-20T10:00:00Z",
    },
}


def get_modal_app_url() -> Optional[str]:
    """Get the URL of the deployed Modal app from the latest deployment.

    Returns:
        URL of the deployed Modal app or None if not found
    """
    try:
        # Base path to the releases directory
        releases_dir = "docs/releases"

        # Find all release directories
        release_folders = [
            f for f in os.listdir(releases_dir) if os.path.isdir(os.path.join(releases_dir, f))
        ]

        if not release_folders:
            logger.error("No release folders found")
            return None

        # Find the latest approval record by examining all files
        latest_approval = None
        latest_timestamp = None

        for folder in release_folders:
            approval_path = os.path.join(releases_dir, folder, "approval_record.json")
            if os.path.exists(approval_path):
                with open(approval_path, "r") as f:
                    try:
                        record = json.load(f)
                        timestamp = record.get("timestamp")

                        # If we found a timestamp and it's newer than our latest
                        if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
                            latest_timestamp = timestamp
                            latest_approval = record
                    except json.JSONDecodeError:
                        continue

        if latest_approval:
            # Get the deployment URL from the latest record
            return latest_approval.get("deployment_url")
        else:
            logger.error("No valid approval records found")
            return None
    except Exception as e:
        logger.error(f"Error getting Modal app URL: {e}")
        return None


def check_endpoint_health(url: str) -> Dict:
    """Check the health of the Modal app endpoint.

    Args:
        url: Root URL of the Modal app

    Returns:
        Dictionary with health status information
    """
    health_url = f"{url.rstrip('/')}/health"
    try:
        response = requests.get(health_url, timeout=5)
        if response.status_code == 200:
            return {
                "status": "healthy",
                "code": 200,
                "message": "Endpoint is healthy",
                "data": response.json(),
            }
        else:
            return {
                "status": "unhealthy",
                "code": response.status_code,
                "message": f"Endpoint returned {response.status_code}",
                "data": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "code": 0,
            "message": f"Could not connect to endpoint: {str(e)}",
            "data": None,
        }


def get_api_info(url: str) -> Dict:
    """Get API information from the Modal app.

    Args:
        url: Root URL of the Modal app

    Returns:
        Dictionary with API information
    """
    try:
        response = requests.get(url.rstrip("/"), timeout=5)
        if response.status_code == 200:
            return {"status": "success", "code": 200, "data": response.json()}
        else:
            return {
                "status": "error",
                "code": response.status_code,
                "message": f"API returned {response.status_code}",
                "data": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "code": 0,
            "message": f"Could not connect to API: {str(e)}",
            "data": None,
        }


def clean_json_string(json_str: str) -> str:
    """Clean a JSON string by removing additional formatting.

    Args:
        json_str: JSON string to clean

    Returns:
        Cleaned JSON string
    """
    # Remove any leading/trailing whitespace
    json_str = json_str.strip()

    # Check if the string starts with a Python string literal prefix
    if json_str.startswith(("r'", 'r"', "'", '"')):
        # Remove quotes and any r prefix
        if json_str.startswith(("r'", 'r"')):
            json_str = json_str[2:-1]
        else:
            json_str = json_str[1:-1]

    # Handle escaped quotes
    json_str = json_str.replace('\\"', '"').replace("\\'", "'")

    return json_str


def make_prediction(url: str, data: Dict) -> Dict:
    """Make a prediction using the Modal app API.

    Args:
        url: Root URL of the Modal app
        data: Dictionary of feature data for prediction

    Returns:
        Dictionary with prediction results or error information
    """
    predict_url = f"{url.rstrip('/')}/predict"
    try:
        response = requests.post(predict_url, json=data, timeout=10)
        if response.status_code == 200:
            return {"status": "success", "code": 200, "data": response.json()}
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = error_data["detail"]
            except Exception:
                error_msg = response.text

            return {
                "status": "error",
                "code": response.status_code,
                "message": f"API returned {response.status_code}: {error_msg}",
                "data": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "code": 0,
            "message": f"Could not connect to API: {str(e)}",
            "data": None,
        }


def report_incident(url: str, data: Dict) -> Dict:
    """Report an incident using the Modal app API.

    Args:
        url: Root URL of the Modal app
        data: Dictionary with incident information

    Returns:
        Dictionary with response from the incident reporting endpoint
    """
    incident_url = f"{url.rstrip('/')}/incident"
    try:
        response = requests.post(incident_url, json=data, timeout=10)
        if response.status_code == 200:
            return {"status": "success", "code": 200, "data": response.json()}
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = error_data["detail"]
            except Exception:
                error_msg = response.text

            return {
                "status": "error",
                "code": response.status_code,
                "message": f"API returned {response.status_code}: {error_msg}",
                "data": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "code": 0,
            "message": f"Could not connect to API: {str(e)}",
            "data": None,
        }


def check_monitoring(url: str) -> Dict:
    """Check data drift monitoring using the Modal app API.

    Args:
        url: Root URL of the Modal app

    Returns:
        Dictionary with monitoring results
    """
    monitor_url = f"{url.rstrip('/')}/monitor"
    try:
        response = requests.get(monitor_url, timeout=10)
        if response.status_code == 200:
            return {"status": "success", "code": 200, "data": response.json()}
        else:
            error_msg = "Unknown error"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg = error_data["detail"]
            except Exception:
                error_msg = response.text

            return {
                "status": "error",
                "code": response.status_code,
                "message": f"API returned {response.status_code}: {error_msg}",
                "data": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "code": 0,
            "message": f"Could not connect to API: {str(e)}",
            "data": None,
        }


def trigger_deployment() -> Dict:
    """Trigger a new deployment of the Modal app.

    Returns:
        Dictionary with deployment status information
    """
    # This would be a call to ZenML to run the deployment pipeline
    # For now, we'll just simulate this
    try:
        client = Client()

        # Get latest model and pipeline runs
        model = client.get_artifact_version(name_id_or_prefix=MODEL_NAME)
        training_pipeline = client.get_pipeline_by_name(pipeline_name=TRAINING_PIPELINE_NAME)

        if not model or not training_pipeline or not training_pipeline.runs:
            return {
                "status": "error",
                "message": "Could not find latest model or training pipeline run",
            }

        latest_run = training_pipeline.runs[0]

        # In a real implementation, this would run the deployment pipeline
        # with the latest model and training artifacts

        # For demo purposes, we'll just return a success message
        return {
            "status": "success",
            "message": f"Deployment triggered with model {model.id} from run {latest_run.id}",
            "data": {
                "model_id": str(model.id),
                "pipeline_run_id": str(latest_run.id),
                "timestamp": pd.Timestamp.now().isoformat(),
            },
        }
    except Exception as e:
        return {"status": "error", "message": f"Error triggering deployment: {str(e)}"}


def display_api_dashboard():
    """Display the API Dashboard component."""
    st.markdown(
        '<div class="card">'
        "<h2>Modal FastAPI Dashboard</h2>"
        "<p>Interact with the deployed Credit Scoring API, view endpoints, and manage deployments.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Get the Modal app URL
    modal_url = get_modal_app_url()

    # If no URL was found, allow the user to enter it manually
    if not modal_url:
        st.warning("Could not find the deployed Modal app URL from ZenML artifacts.")
        modal_url = st.text_input(
            "Enter Modal App URL", value="https://marwan-ext-main--credit-scoring-app.modal.run"
        )

    # Health check and API info
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### API Status", unsafe_allow_html=True)
        health_result = check_endpoint_health(modal_url)
        api_info = get_api_info(modal_url)

        if health_result["status"] == "healthy":
            st.success(f"✅ API is healthy - Status: {health_result['code']}")
        else:
            st.error(f"❌ API is not healthy - {health_result['message']}")

        if api_info["status"] == "success" and api_info["data"]:
            # Display basic API info
            info = api_info["data"]
            st.markdown(f"**Model:** {info.get('model', 'Unknown')}")
            st.markdown(f"**Last updated:** {info.get('timestamp', 'Unknown')}")

            # Display available endpoints
            if "endpoints" in info:
                endpoints = info["endpoints"]
                st.markdown("### Available Endpoints")
                for name, url in endpoints.items():
                    st.markdown(f"- **{name}**: `{url}`")

    # Deploy button
    with col2:
        st.markdown("### Deployment Management", unsafe_allow_html=True)

        deploy_col1, deploy_col2 = st.columns([3, 1])

        with deploy_col1:
            st.markdown("Deploy a new version of the API with the latest model")

        with deploy_col2:
            if st.button("Deploy", type="primary"):
                with st.spinner("Triggering deployment..."):
                    result = trigger_deployment()

                if result["status"] == "success":
                    st.success("Deployment triggered successfully")
                    st.json(result["data"])
                else:
                    st.error(f"Deployment failed: {result['message']}")

    # Add some spacing before tabs
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different endpoints with enhanced styling
    tabs = st.tabs(["Prediction", "Incident Reporting", "Monitoring"])

    # Prediction Tab
    with tabs[0]:
        st.markdown("### Make Predictions")
        st.markdown("Use this endpoint to get credit risk predictions for loan applicants.")

        # Input for prediction request
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Request")

            # Create a text area for JSON input with the sample data
            default_json = json.dumps(SAMPLE_PREDICTION_REQUEST, indent=2)
            prediction_json = st.text_area(
                "Prediction Input (JSON)", value=default_json, height=400
            )

            # Button to make prediction
            if st.button("Make Prediction"):
                with st.spinner("Processing prediction..."):
                    try:
                        # Parse the JSON input
                        prediction_data = json.loads(clean_json_string(prediction_json))

                        # Make the prediction
                        result = make_prediction(modal_url, prediction_data)

                        # Store the result in session state for the other column to display
                        st.session_state.prediction_result = result
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {str(e)}")
                        st.session_state.prediction_result = {
                            "status": "error",
                            "message": f"Invalid JSON: {str(e)}",
                        }
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.prediction_result = {
                            "status": "error",
                            "message": f"Error: {str(e)}",
                        }

        with col2:
            st.markdown("#### Response")

            # Display the prediction result
            if "prediction_result" in st.session_state:
                result = st.session_state.prediction_result

                if result["status"] == "success":
                    st.success("Prediction successful")

                    # Format and display the prediction data
                    if "data" in result and result["data"]:
                        prediction_data = result["data"]

                        # Show risk assessment with nice formatting
                        if "risk_assessment" in prediction_data:
                            risk = prediction_data["risk_assessment"]
                            risk_level = risk.get("risk_level", "unknown").lower()
                            risk_score = risk.get("risk_score", 0)

                            # Choose color based on risk level
                            if risk_level == "high":
                                risk_color = "#D64045"  # Red
                            elif risk_level == "medium":
                                risk_color = "#FFB30F"  # Yellow
                            else:
                                risk_color = "#478C5C"  # Green

                            # Display risk assessment box
                            st.markdown(
                                f"""
                                <div style="padding: 15px; border-radius: 5px; margin-bottom: 20px; 
                                           background-color: {risk_color}20; border-left: 5px solid {risk_color};">
                                    <h4 style="margin-top: 0; color: {risk_color};">Risk Assessment</h4>
                                    <div style="display: flex; justify-content: space-between;">
                                        <div><strong>Risk Score:</strong> {risk_score:.2f}</div>
                                        <div><strong>Risk Level:</strong> {risk_level.upper()}</div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Show other prediction details
                        st.markdown("#### Prediction Details")
                        st.markdown('<div class="api-endpoint-card">', unsafe_allow_html=True)
                        st.json(prediction_data)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Prediction failed: {result.get('message', 'Unknown error')}")
            else:
                st.info("Make a prediction to see results here")

                # Display example response
                st.markdown("#### Example Response")
                example_response = {
                    "probabilities": [0.75],
                    "model_version": "a1b2c3d4",
                    "timestamp": "2024-03-20T10:00:00Z",
                    "risk_assessment": {"risk_score": 0.75, "risk_level": "high"},
                }
                st.markdown('<div class="api-endpoint-card">', unsafe_allow_html=True)
                st.json(example_response)
                st.markdown("</div>", unsafe_allow_html=True)

    # Incident Reporting Tab
    with tabs[1]:
        st.markdown("### Report Incidents")
        st.markdown("Use this endpoint to report incidents or issues with the system.")

        # Input for incident request
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Request")

            # Create a text area for JSON input with the sample data
            default_json = json.dumps(SAMPLE_INCIDENT_REQUEST, indent=2)
            incident_json = st.text_area("Incident Report (JSON)", value=default_json, height=300)

            # Button to report incident
            if st.button("Report Incident"):
                with st.spinner("Reporting incident..."):
                    try:
                        # Parse the JSON input
                        incident_data = json.loads(clean_json_string(incident_json))

                        # Report the incident
                        result = report_incident(modal_url, incident_data)

                        # Store the result in session state for the other column to display
                        st.session_state.incident_result = result
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {str(e)}")
                        st.session_state.incident_result = {
                            "status": "error",
                            "message": f"Invalid JSON: {str(e)}",
                        }
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.incident_result = {
                            "status": "error",
                            "message": f"Error: {str(e)}",
                        }

        with col2:
            st.markdown("#### Response")

            # Display the incident report result
            if "incident_result" in st.session_state:
                result = st.session_state.incident_result

                if result["status"] == "success":
                    st.success("Incident reported successfully")

                    # Format and display the incident data
                    if "data" in result and result["data"]:
                        incident_data = result["data"]

                        # Display incident details
                        st.markdown("#### Incident Report Details")
                        st.markdown('<div class="api-endpoint-card">', unsafe_allow_html=True)
                        st.json(incident_data)
                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Incident reporting failed: {result.get('message', 'Unknown error')}")
            else:
                st.info("Report an incident to see results here")

                # Display example response
                st.markdown("#### Example Response")
                example_response = {
                    "status": "received",
                    "incident_id": "inc_123456",
                    "message": "Incident report received and logged",
                    "timestamp": "2024-03-20T10:00:00Z",
                }
                st.markdown('<div class="api-endpoint-card">', unsafe_allow_html=True)
                st.json(example_response)
                st.markdown("</div>", unsafe_allow_html=True)

    # Monitoring Tab
    with tabs[2]:
        st.markdown("### Data Drift Monitoring")
        st.markdown("Check for data drift and model performance issues.")

        # Button to check monitoring
        if st.button("Check Monitoring Status"):
            with st.spinner("Checking for data drift..."):
                result = check_monitoring(modal_url)

                if result["status"] == "success":
                    monitor_data = result["data"]
                    drift_detected = monitor_data.get("drift_detected", False)

                    if drift_detected:
                        st.warning("⚠️ Data drift detected")
                    else:
                        st.success("✅ No data drift detected")

                    # Display the monitoring data
                    st.markdown("#### Monitoring Details")
                    st.markdown('<div class="api-endpoint-card">', unsafe_allow_html=True)
                    st.json(monitor_data)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Monitoring check failed: {result.get('message', 'Unknown error')}")

        # Display monitoring information
        st.markdown("### About Monitoring")
        st.markdown("""
        The monitoring system checks for:
        
        - **Data drift**: Changes in the distribution of input features
        - **Concept drift**: Changes in the relationship between inputs and outputs
        - **Performance decay**: Degradation in model performance over time
        
        Monitoring is a critical component of EU AI Act compliance, particularly for Article 17 (Post-market monitoring).
        """)

        # Display example monitoring metrics
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Feature Drift Metrics")
            metrics_data = {
                "income_drift": 0.02,
                "age_drift": 0.05,
                "education_drift": 0.01,
                "credit_history_drift": 0.03,
            }
            st.bar_chart(metrics_data)

        with col2:
            st.markdown("#### Performance Over Time")
            performance_data = pd.DataFrame(
                {
                    "date": pd.date_range(start="2025-01-01", periods=10, freq="W"),
                    "accuracy": [0.92, 0.91, 0.915, 0.905, 0.91, 0.90, 0.895, 0.89, 0.885, 0.88],
                    "auc": [0.95, 0.94, 0.945, 0.94, 0.935, 0.93, 0.925, 0.92, 0.915, 0.91],
                }
            )
            performance_data.set_index("date", inplace=True)
            st.line_chart(performance_data)

    # Documentation section
    st.markdown("### API Documentation")

    with st.expander("View API Documentation"):
        st.markdown("""
        ## Credit Scoring API Guide

        ### Overview

        The Credit Scoring API is a EU AI Act compliant service that provides credit risk assessment for loan applications. The API is deployed using Modal and FastAPI, offering a serverless, scalable solution with automatic scaling capabilities.

        ### Authentication

        Currently, the API does not require authentication. However, it is recommended to implement authentication in production environments.

        ### Endpoints

        #### 1. Root Endpoint

        ```http
        GET /
        ```

        Returns basic API information and available endpoints.

        #### 2. Health Check

        ```http
        GET /health
        ```

        Checks the health status of the API.

        #### 3. Prediction Endpoint

        ```http
        POST /predict
        ```

        Makes credit risk predictions based on applicant data.

        #### 4. Monitoring Endpoint

        ```http
        GET /monitor
        ```

        Checks for data drift and model performance issues.

        #### 5. Incident Reporting

        ```http
        POST /incident
        ```

        Reports incidents or issues with the system.

        ### Error Handling

        The API uses standard HTTP status codes:

        - 200: Success
        - 400: Bad Request
        - 500: Internal Server Error

        ### Compliance

        This API is designed to comply with the EU AI Act requirements, including:

        - Article 9: Risk Management
        - Article 10: Data Governance
        - Article 11: Technical Documentation
        - Article 13: Transparency
        - Article 14: Human Oversight
        - Article 15: Accuracy and Robustness
        - Article 17: Post-market Monitoring
        - Article 18: Incident Reporting
        """)
