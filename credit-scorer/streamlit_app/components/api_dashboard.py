"""API Dashboard component for the Modal FastAPI app - Simple Guide Version."""

import json
import logging
import os
from typing import Optional

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Sample requests for documentation
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
    """Get the URL of the deployed Modal app from the latest deployment."""
    try:
        releases_dir = "docs/releases"
        release_folders = [
            f for f in os.listdir(releases_dir)
            if os.path.isdir(os.path.join(releases_dir, f))
        ]

        if not release_folders:
            return None

        latest_approval = None
        latest_timestamp = None

        for folder in release_folders:
            approval_path = os.path.join(releases_dir, folder, "approval_record.json")
            if os.path.exists(approval_path):
                with open(approval_path, "r") as f:
                    try:
                        record = json.load(f)
                        timestamp = record.get("timestamp")
                        if timestamp and (latest_timestamp is None or timestamp > latest_timestamp):
                            latest_timestamp = timestamp
                            latest_approval = record
                    except json.JSONDecodeError:
                        continue

        return latest_approval.get("deployment_url") if latest_approval else None
    except Exception as e:
        logger.error(f"Error getting Modal app URL: {e}")
        return None


def display_api_dashboard():
    """Display the Simple API Dashboard component."""
    st.markdown(
        '<div class="card">'
        "<h2>🚀 Credit Scoring API Guide</h2>"
        "<p>Simple guide to using the Credit Scoring API endpoints.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Get and display API URL
    modal_url = get_modal_app_url()
    
    if not modal_url:
        modal_url = st.text_input(
            "API Base URL",
            value="https://marwan-ext-main--credit-scoring-app.modal.run",
            help="Enter your deployed Modal API URL"
        )

    if modal_url:
        st.markdown("### 🌐 API Base URL")
        st.code(modal_url, language="text")

    # Quick reference
    st.markdown("### 📋 Available Endpoints")
    
    endpoints_data = {
        "Endpoint": ["/", "/health", "/predict", "/monitor", "/incident"],
        "Method": ["GET", "GET", "POST", "GET", "POST"],
        "Purpose": [
            "API Information",
            "Health Check", 
            "Make Predictions",
            "Check Data Drift",
            "Report Issues"
        ]
    }
    
    df = pd.DataFrame(endpoints_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Main tabs
    tabs = st.tabs(["🔮 Predictions", "📊 Monitoring", "🚨 Incidents"])

    # Prediction Tab
    with tabs[0]:
        st.markdown("## Prediction Endpoint")
        st.markdown(f"**URL:** `POST {modal_url}/predict`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Request Body")
            st.code(json.dumps(SAMPLE_PREDICTION_REQUEST, indent=2), language="json")
            
        with col2:
            st.markdown("#### Response")
            example_response = {
                "probabilities": [0.75],
                "model_version": "a1b2c3d4",
                "timestamp": "2024-03-20T10:00:00Z",
                "risk_assessment": {
                    "risk_score": 0.75,
                    "risk_level": "high",
                },
            }
            st.code(json.dumps(example_response, indent=2), language="json")

        # Simple examples
        st.markdown("#### Python Example")
        python_example = f"""import requests

url = "{modal_url}/predict"
data = {json.dumps(SAMPLE_PREDICTION_REQUEST)}

response = requests.post(url, json=data)
result = response.json()

print(f"Risk Score: {{result['risk_assessment']['risk_score']}}")"""
        st.code(python_example, language="python")

        st.markdown("#### cURL Example")
        curl_example = f"""curl -X POST "{modal_url}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(SAMPLE_PREDICTION_REQUEST)}'"""
        st.code(curl_example, language="bash")

    # Monitoring Tab
    with tabs[1]:
        st.markdown("## Monitoring Endpoints")
        
        # Health Check
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Health Check")
            st.markdown(f"**URL:** `GET {modal_url}/health`")
            
            health_response = {
                "status": "healthy",
                "timestamp": "2024-03-20T10:00:00Z"
            }
            st.code(json.dumps(health_response, indent=2), language="json")
            
        with col2:
            st.markdown("#### Data Drift Monitoring")
            st.markdown(f"**URL:** `GET {modal_url}/monitor`")
            
            monitor_response = {
                "drift_detected": False,
                "drift_score": 0.05,
                "model_performance": {
                    "accuracy": 0.92,
                    "auc": 0.95
                },
                "last_check": "2024-03-20T10:00:00Z"
            }
            st.code(json.dumps(monitor_response, indent=2), language="json")

        # Sample monitoring charts
        st.markdown("#### Sample Monitoring Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Drift Scores by Feature**")
            drift_data = {
                "income": 0.02,
                "age": 0.05,
                "education": 0.01,
                "credit_history": 0.03,
            }
            st.bar_chart(drift_data)
            
        with col2:
            st.markdown("**Model Performance Over Time**")
            performance_data = pd.DataFrame({
                "Week": range(1, 11),
                "Accuracy": [0.92, 0.91, 0.915, 0.905, 0.91, 0.90, 0.895, 0.89, 0.885, 0.88],
            })
            st.line_chart(performance_data.set_index("Week"))

    # Incidents Tab
    with tabs[2]:
        st.markdown("## Incident Reporting")
        st.markdown(f"**URL:** `POST {modal_url}/incident`")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Request Body")
            st.code(json.dumps(SAMPLE_INCIDENT_REQUEST, indent=2), language="json")
            
            st.markdown("**Severity Levels:**")
            st.markdown("- `low` - Minor issues")
            st.markdown("- `medium` - Moderate problems") 
            st.markdown("- `high` - Serious issues")
            st.markdown("- `critical` - System failures")
            
        with col2:
            st.markdown("#### Response")
            incident_response = {
                "status": "received",
                "incident_id": "inc_123456",
                "message": "Incident report received and logged",
                "timestamp": "2024-03-20T10:00:00Z",
            }
            st.code(json.dumps(incident_response, indent=2), language="json")

        st.markdown("#### Python Example")
        incident_example = f"""import requests

incident_data = {{
    "severity": "medium",
    "description": "API response time degraded",
    "data": {{"response_time_ms": 5000}}
}}

response = requests.post("{modal_url}/incident", json=incident_data)
result = response.json()
print(f"Incident ID: {{result['incident_id']}}")"""
        st.code(incident_example, language="python")

    # Simple error reference
    st.markdown("---")
    st.markdown("### ❌ Common HTTP Response Codes")
    
    error_data = {
        "Code": ["200", "400", "422", "500"],
        "Status": ["Success", "Bad Request", "Validation Error", "Server Error"],
        "Description": [
            "Request completed successfully",
            "Invalid request format", 
            "Data validation failed",
            "Internal server error"
        ]
    }
    
    error_df = pd.DataFrame(error_data)
    st.dataframe(error_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("---")
    st.info("💡 **Need help?** Check the endpoint responses for detailed error messages.")