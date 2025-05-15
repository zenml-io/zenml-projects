from typing import Any, Dict, Optional, Tuple

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from steps.data_loader import load_data
from steps.data_preprocessor import preprocess_data
from steps.data_validator import validate_data
from steps.predictor import make_predictions
from typing_extensions import Annotated
from zenml import pipeline
from zenml.types import HTMLString


@pipeline(name="retail_forecasting_inference_pipeline")
def inference_pipeline(
    model: Optional[TemporalFusionTransformer] = None,
    training_dataset: Optional[TimeSeriesDataSet] = None,
    forecast_horizon: int = 14,
) -> Tuple[
    Annotated[Dict[str, Any], "forecast_data"],
    Annotated[bytes, "forecast_plot"],
    Annotated[HTMLString, "forecast_visualization"],
]:
    """
    Pipeline to make retail demand forecasts using a trained TFT model.

    Steps:
    1. Load sales and calendar data
    2. Validate data quality
    3. Preprocess data
    4. Make forecasts using a trained model

    Args:
        model: Optional trained model. If None, a simple naive forecast model will be used
        training_dataset: Optional training dataset required for TFT model forecasting
        forecast_horizon: Number of days to forecast into the future

    Returns:
        forecast_data: Dictionary containing forecast data
        forecast_plot: Bytes of the forecast plot image
        forecast_visualization: HTML visualization of forecast results
    """
    # Load data
    sales_data, calendar_data = load_data()

    # Validate data
    sales_data_validated, calendar_data_validated = validate_data(
        sales_data=sales_data, calendar_data=calendar_data
    )

    # Preprocess data
    train_data, val_data, test_data = preprocess_data(
        sales_data=sales_data_validated, calendar_data=calendar_data_validated
    )

    # Make predictions
    forecast_data, forecast_plot, _, _, _, viz = make_predictions(
        model=model,
        training_dataset=training_dataset,
        test_data=test_data,
        forecast_horizon=forecast_horizon,
    )

    # Return the forecast data and visualizations
    return forecast_data, forecast_plot, viz
