from zenml import pipeline
from steps.data_loader import load_data
from steps.data_validator import validate_data
from steps.data_preprocessor import preprocess_data
from steps.predictor import make_predictions
from typing import Dict, Any, Optional, Tuple
from typing_extensions import Annotated
from zenml.types import HTMLString


@pipeline(name="retail_forecasting_inference_pipeline")
def inference_pipeline(
    model_artifacts: Optional[Dict[str, Any]] = None,
    forecast_horizon: int = 14,
) -> Tuple[
    Annotated[Dict[str, Any], "forecast_data"],
    Annotated[bytes, "forecast_plot"],
    Annotated[Dict[str, Any], "sample_forecast"],
    Annotated[int, "forecast_horizon"],
    Annotated[str, "method"],
    Annotated[HTMLString, "forecast_visualization"]
]:
    """
    Pipeline to make retail demand forecasts using a trained TFT model.
    
    Steps:
    1. Load sales and calendar data
    2. Validate data quality
    3. Preprocess data 
    4. Make forecasts using a trained model
    
    Args:
        model_artifacts: Optional dictionary containing the trained model and required artifacts
                        If None, a simple naive forecast model will be used
        forecast_horizon: Number of days to forecast into the future
        
    Returns:
        Tuple containing:
            - forecast_data: Dictionary containing forecast data
            - forecast_plot: Bytes of the forecast plot image
            - sample_forecast: Dictionary with sample forecasts
            - forecast_horizon: Number of days in the forecast
            - method: Name of the forecasting method used
            - forecast_visualization: HTML visualization of forecast results
    """
    # Load data
    data = load_data()
    
    # Validate data
    validated_data = validate_data(data=data)
    
    # Preprocess data (with train_test_split needed for forecasting)
    processed_data = preprocess_data(data=validated_data)
    
    # Make predictions with model (or naive forecast if no model provided)
    return make_predictions(
        model_artifacts=model_artifacts,
        processed_data=processed_data,
        forecast_horizon=forecast_horizon
    )
