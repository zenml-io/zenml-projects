from typing import Dict, Tuple

from steps.data_loader import load_data
from steps.data_preprocessor import preprocess_data
from steps.model_evaluator import evaluate_models
from steps.model_trainer import train_model
from steps.predictor import generate_forecasts
from typing_extensions import Annotated
from zenml import pipeline
from zenml.types import HTMLString


@pipeline(name="retail_forecast_pipeline")
def training_pipeline() -> Tuple[
    Annotated[Dict[str, float], "model_metrics"],
    Annotated[HTMLString, "evaluation_report"],
    Annotated[HTMLString, "forecast_dashboard"],
]:
    """
    Simple retail forecasting pipeline using Prophet.

    Steps:
    1. Load sales data
    2. Preprocess data for Prophet
    3. Train Prophet models (one per store-item combination)
    4. Evaluate model performance on test data
    5. Generate forecasts for future periods

    Args:
        test_size: Proportion of data to use for testing
        forecast_periods: Number of days to forecast into the future
        weekly_seasonality: Whether to include weekly seasonality in the model

    Returns:
        model_metrics: Dictionary of performance metrics
        evaluation_report: HTML report of model evaluation
        forecast_dashboard: HTML dashboard of forecasts
    """
    # Load synthetic retail data
    sales_data = load_data()

    # Preprocess data for Prophet
    train_data_dict, test_data_dict, series_ids = preprocess_data(
        sales_data=sales_data
    )

    # Train Prophet models for each series
    models = train_model(
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    # Evaluate models
    metrics, evaluation_report = evaluate_models(
        models=models, test_data_dict=test_data_dict, series_ids=series_ids
    )

    # Generate forecasts
    _, _, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    return metrics, evaluation_report, forecast_dashboard
