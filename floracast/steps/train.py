"""
Model training step for FloraCast.
"""

from typing import Annotated

import torch
from darts import TimeSeries
from darts.models import TFTModel
from materializers.tft_materializer import (
    TFTModelMaterializer,
)  # Import for explicit usage
from zenml import step
from zenml.logger import get_logger

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
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
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
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization

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
        "optimizer_kwargs": {
            "lr": learning_rate,
            "weight_decay": weight_decay,
        },
        "pl_trainer_kwargs": {
            "enable_progress_bar": enable_progress_bar,
            "enable_model_summary": enable_model_summary,
            "precision": "32-true",  # Use 32-bit precision for better hardware compatibility
            "gradient_clip_val": 1.0,  # Standard gradient clipping
            "gradient_clip_algorithm": "norm",  # Clip by norm
            "detect_anomaly": True,  # Detect NaN/inf in loss
            "max_epochs": n_epochs,
            "check_val_every_n_epoch": 1,  # Validate every epoch
            "accelerator": "cpu",  # Force CPU to avoid MPS issues
        },
    }

    logger.info(f"Training TFT model with params: {model_params}")

    # Initialize TFT model
    model = TFTModel(**model_params)

    # Initialize model weights with Xavier/Glorot initialization for stability
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    # Apply weight initialization
    if hasattr(model, "model") and model.model is not None:
        model.model.apply(init_weights)
        logger.info("Applied Xavier weight initialization to model")

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
