from zenml import pipeline
from steps.data_loader import load_data
from steps.data_validator import validate_data
from steps.data_preprocessor import preprocess_data
from steps.model_trainer import train_model
from steps.model_evaluator import evaluate_model
from typing import Dict, Any, Tuple
from typing_extensions import Annotated


@pipeline(name="retail_forecasting_training_pipeline")
def training_pipeline(
    forecast_horizon: int = 14,
    hidden_size: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_encoder_length: int = 30,
    batch_size: int = 64,
    max_epochs: int = 10,  # Reduced for faster execution, increase for better results
) -> Tuple[
    Annotated[float, "mae"],
    Annotated[float, "rmse"],
    Annotated[float, "smape"],
    Annotated[float, "mape"]
]:
    """
    Pipeline to train a retail demand forecasting model.

    Steps:
    1. Load sales and calendar data
    2. Validate data quality
    3. Preprocess data (feature engineering, encode categorical variables)
    4. Train a Temporal Fusion Transformer model
    5. Evaluate model performance on test data

    Returns:
        Tuple containing key metrics:
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - SMAPE: Symmetric Mean Absolute Percentage Error
        - MAPE: Mean Absolute Percentage Error
    """
    # Load data
    data = load_data()

    # Validate data
    validated_data = validate_data(data=data)

    # Preprocess data
    processed_data = preprocess_data(data=validated_data)

    # Train model
    model_artifacts = train_model(
        processed_data=processed_data,
        forecast_horizon=forecast_horizon,
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        max_encoder_length=max_encoder_length,
        batch_size=batch_size,
        max_epochs=max_epochs,
    )

    # Evaluate model
    (
        mae, 
        rmse, 
        smape, 
        mape, 
        error_plot,
        worst_series,
        store_errors,
        item_errors,
        date_errors,
        model,
        evaluation_visualization
    ) = evaluate_model(
        model_artifacts=model_artifacts, processed_data=processed_data
    )

    # Return just the key metrics
    return mae, rmse, smape, mape
