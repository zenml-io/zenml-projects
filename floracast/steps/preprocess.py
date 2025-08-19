"""
Data preprocessing steps for FloraCast.
"""

from typing import Tuple, Annotated
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from zenml import step
from zenml.logger import get_logger
from materializers.timeseries_materializer import DartsTimeSeriesMaterializer


logger = get_logger(__name__)


@step(
    output_materializers={
        "train_series": DartsTimeSeriesMaterializer,
        "val_series": DartsTimeSeriesMaterializer,
    }
)
def preprocess_for_training(
    df: pd.DataFrame,
    datetime_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    val_ratio: float = 0.2,
) -> Tuple[
    Annotated[TimeSeries, "train_series"], Annotated[TimeSeries, "val_series"]
]:
    """
    Preprocess data for training - splits into train/val sets.

    Args:
        df: Raw DataFrame with datetime and target columns
        datetime_col: Name of datetime column
        target_col: Name of target column
        freq: Frequency string for time series
        val_ratio: Ratio of data to use for validation

    Returns:
        Tuple of (train_series, val_series)
    """
    logger.info(
        "🎯 TRAINING MODE: Converting to Darts TimeSeries and splitting data"
    )
    logger.info(f"Converting to TimeSeries with frequency: {freq}")

    # Create Darts TimeSeries (keeping it simple - just target variable)
    series = TimeSeries.from_dataframe(
        df, time_col=datetime_col, value_cols=target_col, freq=freq
    )

    logger.info(f"Created TimeSeries with {len(series)} points")
    logger.info(f"Series range: {series.start_time()} to {series.end_time()}")

    # Split into train and validation BEFORE scaling
    split_point = int(len(series) * (1 - val_ratio))
    train_series_raw = series[:split_point]
    val_series_raw = series[split_point:]

    # Apply normalization using training data statistics
    scaler = Scaler()
    logger.info(
        "Fitting scaler on training data and applying to both train/val"
    )

    # Fit scaler only on training data to prevent data leakage
    train_series = scaler.fit_transform(train_series_raw)
    val_series = scaler.transform(val_series_raw)

    # Log scaling statistics
    train_mean = train_series_raw.pd_dataframe().iloc[:, 0].mean()
    train_std = train_series_raw.pd_dataframe().iloc[:, 0].std()
    logger.info(
        f"Scaling stats - Mean: {train_mean:.2f}, Std: {train_std:.2f}"
    )

    logger.info(
        f"Train series: {len(train_series)} points ({train_series.start_time()} to {train_series.end_time()})"
    )
    logger.info(
        f"Validation series: {len(val_series)} points ({val_series.start_time()} to {val_series.end_time()})"
    )

    return train_series, val_series


@step(
    output_materializers={
        "full_series": DartsTimeSeriesMaterializer,
    }
)
def preprocess_for_inference(
    df: pd.DataFrame,
    datetime_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
) -> Annotated[TimeSeries, "full_series"]:
    """
    Preprocess data for inference - returns full series as context.

    Args:
        df: Raw DataFrame with datetime and target columns
        datetime_col: Name of datetime column
        target_col: Name of target column
        freq: Frequency string for time series

    Returns:
        Full TimeSeries for forecasting context
    """
    logger.info(
        "🔮 INFERENCE MODE: Converting to Darts TimeSeries for forecasting context"
    )
    logger.info(f"Converting to TimeSeries with frequency: {freq}")

    # Create Darts TimeSeries
    series = TimeSeries.from_dataframe(
        df, time_col=datetime_col, value_cols=target_col, freq=freq
    )

    logger.info(f"Created TimeSeries with {len(series)} points")
    logger.info(f"Series range: {series.start_time()} to {series.end_time()}")
    logger.info("📈 Using full historical data as context for forecasting")
    logger.info(f"Full series: {len(series)} points for forecasting context")

    return series
