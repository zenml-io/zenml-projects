from zenml import step
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from typing import Dict, Any


@step
def train_model(
    processed_data: dict,
    forecast_horizon: int = 14,  # Forecast horizon in days
    hidden_size: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    max_encoder_length: int = 30,  # Look-back window
    batch_size: int = 64,
    max_epochs: int = 50,
) -> Dict[str, Any]:
    """Train a Temporal Fusion Transformer (TFT) model for retail forecasting."""
    train_df = processed_data["train"]
    val_df = processed_data["val"]
    features = processed_data["features"]

    # Check if we have a GPU available
    if torch.cuda.is_available():
        accelerator = "gpu"
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = "cpu"
        print("No GPU available, training on CPU")

    # Create TimeSeriesDataSet for PyTorch Forecasting
    # This handles all the complexity of preparing time series data for deep learning
    training_cutoff = train_df["time_idx"].max()

    # Define the dataset
    training = TimeSeriesDataSet(
        data=train_df,
        time_idx="time_idx",
        target="sales",  # Target variable to forecast
        group_ids=[
            "series_id"
        ],  # Each store-item combo is a separate time series
        max_encoder_length=max_encoder_length,  # Look-back window
        max_prediction_length=forecast_horizon,  # Forecast horizon
        static_categoricals=[
            "store_encoded",
            "item_encoded",
        ],  # Store and item are static (unchanging) features
        static_reals=[],  # No static real features yet
        time_varying_known_categoricals=[
            "day_of_week",
            "month",
        ],  # Features we know in the future
        time_varying_known_reals=[
            "is_holiday",
            "is_promo",
            "is_weekend",
        ],  # Known numeric features
        time_varying_unknown_reals=["sales"]
        + [
            feat for feat in features["continuous"] if feat in train_df.columns
        ],  # Features we don't know in advance
        variable_groups={},  # No variable groups needed
        target_normalizer=GroupNormalizer(
            groups=["series_id"],
            transformation="softplus",  # Ensures forecasts are positive
        ),  # Scale each time series independently
        add_relative_time_idx=True,  # Add relative time index as a feature
        add_target_scales=True,  # Add target scale to help denormalize predictions
        add_encoder_length=True,  # Add encoder length to help sample weights
    )

    # Create validation dataset from validation dataframe
    validation = TimeSeriesDataSet.from_dataset(
        training, val_df, stop_randomization=True
    )

    # Create data loaders for training
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    # Define the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,  # Size of hidden layers
        attention_head_size=4,  # Number of attention heads
        dropout=dropout,  # Dropout rate
        hidden_continuous_size=hidden_size,  # Size for processing continuous variables
        learning_rate=learning_rate,  # Learning rate
        optimizer="ranger",  # RAdam with lookahead
        loss=QuantileLoss(),  # Quantile loss for probabilistic forecasting
        log_interval=10,  # Logging interval
        reduce_on_plateau_patience=3,  # Patience for learning rate reduction
    )

    print(f"Number of parameters in model: {tft.size() / 1e3:.1f}k")

    # Create PyTorch Lightning trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=True,
        mode="min",
    )

    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        gradient_clip_val=0.1,
        limit_train_batches=50
        if max_epochs <= 10
        else 200,  # For demonstration, use fewer batches for quick runs
        callbacks=[early_stop_callback, lr_logger],
        enable_model_summary=True,
    )

    # Fit the model
    print("Training TFT model...")
    trainer.fit(
        tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )

    # Return the trained model and dataset directly as artifacts
    # No need to save to disk as ZenML will handle the storage
    return {"model": tft, "training_dataset": training, "model_type": "tft"}
