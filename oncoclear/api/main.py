import datetime
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from zenml import log_metadata
from zenml.client import Client
from zenml.logger import get_logger as zenml_get_logger

# Configure logging
# Use ZenML logger for consistency if desired, or standard Python logging
logger = zenml_get_logger(__name__)
# Basic logging configuration (adjust as needed)
# logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
# logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_STAGE = os.getenv("MODEL_STAGE", "production")  # Default to production
MODEL_ARTIFACT_NAME = os.getenv(
    "MODEL_ARTIFACT_NAME", "sklearn_classifier"
)  # Updated default artifact name
PREPROCESS_PIPELINE_NAME = os.getenv(
    "PREPROCESS_PIPELINE_NAME", "preprocess_pipeline"
)  # Added preprocessing pipeline artifact name
# ZENML_STORE_URL and ZENML_STORE_API_KEY are automatically picked up by Client if set

if not MODEL_NAME:
    logger.error("Environment variable MODEL_NAME is not set.")
    # Or raise an exception to prevent startup
    raise ValueError("MODEL_NAME must be set via environment variable")

# --- Global Variables ---
# This dictionary will hold application state, like the loaded model
app_state = {"model": None, "preprocess_pipeline": None}


# --- Helper function to convert numpy arrays to Python native types ---
def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


# --- Pydantic Models ---
class FeaturesPayload(BaseModel):
    # Expecting a single instance/row for prediction
    # Adjust structure based on your model's expected input (e.g., dict, list of lists)
    features: List[Any]  # Using Any for flexibility, refine if possible

    # Example for sklearn models often expecting a list of lists (even for one sample)
    # features: List[List[float]]
    # Example for named features
    # feature_dict: Dict[str, float]


class PredictionResponse(BaseModel):
    prediction: Union[int, float, str, bool, List[Any], Dict[str, Any]]


class DebugResponse(BaseModel):
    zenml_url: str
    api_key_provided: bool
    connection_test: str
    error_details: str = None


# --- FastAPI Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("FastAPI application starting up...")

    # Debug: Log environment variables (masking sensitive ones)
    zenml_url = os.getenv("ZENML_STORE_URL", "Not set")
    zenml_key = os.getenv("ZENML_STORE_API_KEY", "Not set")
    if zenml_key != "Not set":
        masked_key = (
            zenml_key[:15] + "..." + zenml_key[-10:]
            if len(zenml_key) > 30
            else "***masked***"
        )
    else:
        masked_key = "Not set"

    logger.info(f"Environment variables:")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"MODEL_STAGE: {MODEL_STAGE}")
    logger.info(f"MODEL_ARTIFACT_NAME: {MODEL_ARTIFACT_NAME}")
    logger.info(f"ZENML_STORE_URL: {zenml_url}")
    logger.info(f"ZENML_STORE_API_KEY: {masked_key}")
    logger.info(
        f"ZENML_CONFIG_DIR: {os.getenv('ZENML_CONFIG_DIR', 'Not set')}"
    )

    try:
        logger.info(
            f"Attempting to load model '{MODEL_NAME}' version '{MODEL_STAGE}'."
        )
        # Ensure environment variables for ZenML connection are set externally
        client = Client()  # Client automatically uses ENV vars if set
        model_version = client.get_model_version(MODEL_NAME, MODEL_STAGE)

        # Load the model artifact using the artifact name from env vars
        artifact_name = MODEL_ARTIFACT_NAME
        logger.info(
            f"Found model version {model_version.name}. Loading model artifact '{artifact_name}'..."
        )
        app_state["model"] = model_version.get_artifact(artifact_name).load()
        logger.info(
            f"Successfully loaded model artifact '{artifact_name}' from {MODEL_NAME}:{model_version.name}"
        )

        # Also load the preprocessing pipeline artifact if available
        preprocess_name = PREPROCESS_PIPELINE_NAME
        try:
            logger.info(
                f"Attempting to load preprocessing pipeline artifact '{preprocess_name}'..."
            )
            app_state["preprocess_pipeline"] = model_version.get_artifact(
                preprocess_name
            ).load()
            logger.info(
                f"Successfully loaded preprocessing pipeline artifact."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load preprocessing pipeline: {e}. Predictions may require pre-processed input."
            )
            app_state["preprocess_pipeline"] = None

    except Exception as e:
        logger.error(
            f"Failed to load model during startup: {e}", exc_info=True
        )
        # Decide if the app should fail to start or continue without a model
        # Option 1: Raise exception to stop startup
        raise RuntimeError(f"Model loading failed: {e}")
        # Option 2: Log error and continue (endpoints needing model will fail)
        # app_state["model"] = None

    yield  # Application runs here

    # Shutdown logic (if any)
    logger.info("FastAPI application shutting down...")
    app_state["model"] = None
    app_state["preprocess_pipeline"] = None


# --- FastAPI App ---
app = FastAPI(
    title="ML Model Deployment API",
    description=f"API for serving the '{MODEL_NAME}' model.",
    version="0.1.0",
    lifespan=lifespan,  # Use the lifespan context manager
)


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    # Optionally add checks for model readiness
    model_ready = app_state.get("model") is not None
    if not model_ready:
        # Service unavailable if model didn't load
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_ready": model_ready}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: FeaturesPayload):
    """
    Make predictions using the loaded model.
    If a preprocessing pipeline is available, it will be applied to the input data.
    """
    model = app_state.get("model")
    preprocess_pipeline = app_state.get("preprocess_pipeline")

    if model is None:
        logger.error("Prediction endpoint called but model is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model not loaded or failed to load."
        )

    try:
        logger.debug(f"Received prediction request: {payload}")

        # Capture request timestamp
        timestamp = datetime.datetime.now().isoformat()

        # Convert input to format expected by model - as a 2D array
        # This handles the input format for scikit-learn models
        data_to_predict = [payload.features]
        logger.debug(f"Input data before preprocessing: {data_to_predict}")

        # Apply preprocessing if available
        if preprocess_pipeline is not None:
            logger.debug("Applying preprocessing pipeline to input data")
            try:
                # Now the preprocessing pipeline should be properly loaded with transform() method
                data_to_predict = preprocess_pipeline.transform(
                    data_to_predict
                )
                logger.debug(f"Data after preprocessing: {data_to_predict}")
            except Exception as e:
                logger.error(f"Error applying preprocessing: {e}")
                # Fall back to using the raw input if preprocessing fails
                logger.warning(
                    "Falling back to raw input without preprocessing"
                )

        # Make prediction with the model
        prediction_result = model.predict(data_to_predict)

        # Extract the first prediction if predict returns an array/list
        prediction_value = (
            prediction_result[0]
            if isinstance(prediction_result, (list, tuple, np.ndarray))
            and len(prediction_result) > 0
            else prediction_result
        )

        # Convert numpy arrays and other non-serializable types to Python native types
        serializable_prediction = convert_to_serializable(prediction_value)

        # Log prediction metadata to ZenML
        try:
            serializable_input = convert_to_serializable(payload.features)

            # Create prediction metadata
            prediction_metadata = {
                "prediction_info": {
                    "timestamp": timestamp,
                    "input": serializable_input,
                    "prediction": serializable_prediction,
                    "model_name": MODEL_NAME,
                    "model_stage": MODEL_STAGE,
                }
            }

            # Log metadata to the model version
            logger.debug(
                f"Logging prediction metadata to model {MODEL_NAME}:{MODEL_STAGE}"
            )
            log_metadata(
                metadata=prediction_metadata,
                model_name=MODEL_NAME,
                model_version=MODEL_STAGE,
            )
            logger.debug("Successfully logged prediction metadata")
        except Exception as log_error:
            # Don't fail the API call if metadata logging fails
            logger.warning(f"Failed to log prediction metadata: {log_error}")

        logger.debug(
            f"Prediction result (serializable): {serializable_prediction}"
        )
        return PredictionResponse(prediction=serializable_prediction)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )


# Optional: Add a root endpoint for basic info
@app.get("/")
async def read_root():
    return {
        "message": f"Welcome to the prediction API for model '{MODEL_NAME}'"
    }


@app.get("/debug", response_model=DebugResponse)
async def debug_connection():
    """
    Debug endpoint to check ZenML connection.
    """
    zenml_url = os.getenv("ZENML_STORE_URL", "Not set")
    api_key = os.getenv("ZENML_STORE_API_KEY", "Not set")
    api_key_provided = api_key != "Not set"

    result = {
        "zenml_url": zenml_url,
        "api_key_provided": api_key_provided,
        "connection_test": "Not attempted",
        "error_details": None,
    }

    if api_key_provided and zenml_url != "Not set":
        try:
            # Try to initialize client and make a simple request
            from zenml.client import Client

            # This creates a Client which should attempt to connect to the server
            client = Client()

            # Try to get the current user, which requires authentication
            try:
                user = client.zen_store.get_user()
                result["connection_test"] = (
                    f"Success - authenticated as {user.name}"
                )
            except Exception as e:
                result["connection_test"] = "Failed - Authentication error"
                result["error_details"] = str(e)

        except Exception as e:
            result["connection_test"] = "Failed - Client initialization error"
            result["error_details"] = str(e)

    return result


# --- Main Execution (for local testing, typically run by Uvicorn) ---
if __name__ == "__main__":
    # This block is mainly for local development/debugging.
    # In production/deployment, Uvicorn runs the app instance directly.
    # Set environment variables locally (e.g., using a .env file and python-dotenv)
    # before running this script for testing.
    logger.info("Running FastAPI app locally with Uvicorn...")
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,  # Enable reload for development convenience
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
