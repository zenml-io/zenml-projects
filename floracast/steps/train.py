"""
Model training step for FloraCast.
"""

from typing import Annotated
import torch
from darts import TimeSeries
from darts.models import TFTModel
from zenml import step
from zenml.logger import get_logger
from materializers.tft_materializer import (
    TFTModelMaterializer,
)  # Import for explicit usage

logger = get_logger(__name__)


@step(output_materializers={"trained_model": TFTModelMaterializer})
def train_model(
    train_series: TimeSeries,
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
    """Train a TFT forecasting model.

    Args:
        train_series: Training time series
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
        Trained TFT model
    """
    # Build TFT model parameters
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
            "precision": "32-true",  # Use 32-bit precision for better hardware compatibility
        },
    }

    logger.info(f"Training TFT model with params: {model_params}")

    # Initialize TFT model
    model = TFTModel(**model_params)
    logger.info(f"Starting TFT training with {len(train_series)} data points")

    # Train the TFT model
    # Get basic stats from the pandas dataframe
    df_stats = train_series.pd_dataframe().iloc[:, 0]
    logger.info(
        f"Train series stats: min={df_stats.min():.2f}, max={df_stats.max():.2f}, mean={df_stats.mean():.2f}"
    )

    model.fit(train_series)
    logger.info("TFT model training completed successfully")

    # Test prediction to verify model works
    test_pred = model.predict(n=1, series=train_series)
    logger.info(f"Test prediction: {test_pred.pd_dataframe().iloc[0, 0]:.2f}")

    return model
