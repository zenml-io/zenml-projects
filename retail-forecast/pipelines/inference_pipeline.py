from zenml import pipeline
from steps.data_loader import load_data
from steps.data_validator import validate_data
from steps.data_preprocessor import preprocess_data
from steps.predictor import make_predictions
from typing import Dict, Any, Optional


@pipeline(name="retail_forecasting_inference_pipeline")
def inference_pipeline(
    forecast_horizon: int = 14,  # Number of days to forecast into the future
    model_artifacts: Optional[
        Dict[str, Any]
    ] = None,  # Model artifacts from the training pipeline
):
    """
    Pipeline to generate retail demand forecasts using a trained model.

    Steps:
    1. Load latest sales and calendar data
    2. Validate data quality
    3. Preprocess data (same feature engineering as training)
    4. Load trained model and generate forecasts

    Args:
        forecast_horizon: Number of days to forecast into the future
        model_artifacts: Model and training dataset artifacts from training pipeline
    """
    # Load data
    data = load_data()

    # Validate data
    validated_data = validate_data(data=data)

    # Preprocess data
    processed_data = preprocess_data(data=validated_data)

    # Generate forecasts using the model artifacts
    # If None, the step will try to use a cached model from previous runs
    forecasts = make_predictions(
        model_artifacts=model_artifacts,
        processed_data=processed_data,
        forecast_horizon=forecast_horizon,
    )

    return forecasts
