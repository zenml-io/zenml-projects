"""
Model training step for FloraCast.
"""

from typing import Annotated
from darts import TimeSeries
from darts.models import TFTModel, ExponentialSmoothing
from zenml import step
from zenml.logger import get_logger
from materializers.tft_materializer import (
    TFTModelMaterializer,
)  # Import for explicit usage

logger = get_logger(__name__)


@step(output_materializers={"trained_model": TFTModelMaterializer})
def train_model(
    train_series: TimeSeries,
    model_name: str = "TFTModel",
    input_chunk_length: int = 30,
    output_chunk_length: int = 7,
    hidden_size: int = 32,
    lstm_layers: int = 1,
    num_attention_heads: int = 2,
    dropout: float = 0.1,
    batch_size: int = 16,
    n_epochs: int = 20,
    random_state: int = 42,
    add_relative_index: bool = True,
    enable_progress_bar: bool = False,
    enable_model_summary: bool = False,
) -> Annotated[TFTModel, "trained_model"]:
    """
    Train a forecasting model.

    Args:
        train_series: Training time series
        model_name: Name of the model class to use
        input_chunk_length: Number of time steps to use as input
        output_chunk_length: Number of time steps to predict
        hidden_size: Size of hidden layers
        lstm_layers: Number of LSTM layers
        num_attention_heads: Number of attention heads
        dropout: Dropout rate
        batch_size: Training batch size
        n_epochs: Number of training epochs
        random_state: Random seed
        add_relative_index: Whether to add relative index
        enable_progress_bar: Whether to show progress bar
        enable_model_summary: Whether to show model summary

    Returns:
        Tuple of (fitted_model, artifact_uri, model_class)
    """
    # Build model parameters dict
    model_params = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "hidden_size": hidden_size,
        "lstm_layers": lstm_layers,
        "num_attention_heads": num_attention_heads,
        "dropout": dropout,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "random_state": random_state,
        "add_relative_index": add_relative_index,
        "pl_trainer_kwargs": {
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": enable_model_summary,
        },
    }

    logger.info(f"Training {model_name} with params: {model_params}")

    # Initialize model based on configuration
    if model_name == "TFTModel":
        model = TFTModel(**model_params)
    elif model_name == "ExponentialSmoothing":
        # Fallback model with simpler parameters
        fallback_params = {
            k: v
            for k, v in model_params.items()
            if k in ["seasonal_periods", "trend", "seasonal"]
        }
        model = ExponentialSmoothing(**fallback_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    logger.info(f"Starting training with {len(train_series)} data points")

    # Train the model
    try:
        logger.info(f"Starting TFT training with {len(train_series)} points")
        # Get basic stats from the pandas dataframe
        df_stats = train_series.pd_dataframe().iloc[:, 0]
        logger.info(
            f"Train series stats: min={df_stats.min():.2f}, max={df_stats.max():.2f}, mean={df_stats.mean():.2f}"
        )

        model.fit(train_series)
        logger.info("TFT Model training completed successfully")

        # Test prediction to verify model works
        test_pred = model.predict(n=1, series=train_series)
        logger.info(
            f"Test prediction: {test_pred.pd_dataframe().iloc[0, 0]:.2f} (should be similar to training data range)"
        )

    except Exception as e:
        logger.error(f"TFT Training failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback

        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Fallback to simpler model with proper parameters
        logger.info("Falling back to ExponentialSmoothing model")
        from darts.models import ExponentialSmoothing

        model = ExponentialSmoothing(
            seasonal_periods=7  # Weekly seasonality
        )
        model.fit(train_series)
        logger.info("Fallback ExponentialSmoothing model training completed")

        # Test fallback prediction (ExponentialSmoothing doesn't need series parameter)
        test_pred = model.predict(n=1)
        logger.info(
            f"Fallback test prediction: {test_pred.pd_dataframe().iloc[0, 0]:.2f}"
        )

    return model
