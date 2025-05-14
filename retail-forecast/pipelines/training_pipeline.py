from zenml import pipeline
from steps.data_loader import load_data
from steps.data_validator import validate_data
from steps.data_preprocessor import preprocess_data
from steps.model_trainer import train_model
from steps.model_evaluator import evaluate_model
from typing import Tuple
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from typing_extensions import Annotated


@pipeline(name="retail_forecasting_training_pipeline")
def training_pipeline(
    forecast_horizon: int = 14,
    hidden_size: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_encoder_length: int = 30,
    batch_size: int = 64,
    max_epochs: int = 50,
) -> Tuple[
    Annotated[TemporalFusionTransformer, "trained_model"],
    Annotated[TimeSeriesDataSet, "training_dataset"]
]:
    """
    Pipeline to train a retail demand forecasting model.

    Steps:
    1. Load sales and calendar data
    2. Validate data quality
    3. Preprocess data (feature engineering, encode categorical variables)
    4. Train a Temporal Fusion Transformer model
    5. Evaluate model performance on test data

    Args:
        forecast_horizon: Number of days to forecast into the future
        hidden_size: Hidden size for the TFT model
        dropout: Dropout rate for the model
        learning_rate: Learning rate for training
        max_encoder_length: Look-back window (in days)
        batch_size: Batch size for training
        max_epochs: Maximum number of training epochs

    Returns:
        trained_model: The trained TFT model
        training_dataset: The dataset configuration for future use
    """
    # Load data
    sales_data, calendar_data = load_data()

    # Validate data
    sales_data_validated, calendar_data_validated = validate_data(
        sales_data=sales_data, 
        calendar_data=calendar_data
    )

    # Preprocess data
    train_data, val_data, test_data = preprocess_data(
        sales_data=sales_data_validated, 
        calendar_data=calendar_data_validated
    )

    # Train model
    model, training_dataset = train_model(
        train_data=train_data,
        val_data=val_data,
        forecast_horizon=forecast_horizon,
        hidden_size=hidden_size,
        dropout=dropout,
        learning_rate=learning_rate,
        max_encoder_length=max_encoder_length,
        batch_size=batch_size,
        max_epochs=max_epochs
    )

    # Evaluate model - metrics are logged via log_metadata
    evaluate_model(
        model=model,
        training_dataset=training_dataset, 
        test_data=test_data
    )

    # Return trained model and dataset for future predictions
    return model, training_dataset
