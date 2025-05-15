from typing import Dict, List, Tuple, Optional

import pandas as pd
from prophet import Prophet
from steps.data_loader import load_data
from steps.data_preprocessor import preprocess_data
from steps.predictor import generate_forecasts
from typing_extensions import Annotated
from zenml import pipeline
from zenml.types import HTMLString
from zenml import get_pipeline_context


@pipeline(name="retail_forecast_inference_pipeline")
def inference_pipeline():
    """
    Pipeline to make retail demand forecasts using trained Prophet models.

    This pipeline is for when you already have trained models and want to
    generate new forecasts without retraining.

    Steps:
    1. Load sales data
    2. Preprocess data
    3. Generate forecasts using provided models or simple baseline models

    Args:
        models: Optional dictionary of trained Prophet models. If None, simple models will be created
        forecast_periods: Number of days to forecast into the future

    Returns:
        combined_forecast: Combined dataframe with all series forecasts
        forecast_dashboard: HTML dashboard with forecast visualizations
    """
    # Load data
    sales_data = load_data()

    # Preprocess data
    train_data_dict, _, series_ids = preprocess_data(
        sales_data=sales_data,
        test_size=0,
    )

    # Generate forecasts
    models = get_pipeline_context().model.get_artifact(
        "trained_prophet_models"
    )
    _, combined_forecast, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    # Return forecast data and dashboard
    return combined_forecast, forecast_dashboard
