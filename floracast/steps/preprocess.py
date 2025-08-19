"""
Data preprocessing step for FloraCast.
"""

from typing import Dict, Any, Tuple, Annotated
import pandas as pd
from darts import TimeSeries
from zenml import step
from zenml.logger import get_logger

from materializers.darts_materializer import DartsTimeSeriesMaterializer

logger = get_logger(__name__)


@step(output_materializers=DartsTimeSeriesMaterializer)
def preprocess(
    df: Annotated[pd.DataFrame, "raw_data"],
    datetime_col: str = "ds",
    target_col: str = "y",
    freq: str = "D",
    val_ratio: float = 0.2
) -> Tuple[
    Annotated[TimeSeries, "train_series"],
    Annotated[TimeSeries, "val_series"]
]:
    """
    Preprocess data and convert to Darts TimeSeries format.
    
    Args:
        df: Raw DataFrame with datetime and target columns
        datetime_col: Name of datetime column
        target_col: Name of target column
        freq: Frequency string for time series
        val_ratio: Ratio of data to use for validation
        
    Returns:
        Tuple of (train_series, val_series)
    """
    
    logger.info(f"Converting to Darts TimeSeries with frequency: {freq}")
    
    # Create Darts TimeSeries
    series = TimeSeries.from_dataframe(
        df,
        time_col=datetime_col,
        value_cols=target_col,
        freq=freq
    )
    
    logger.info(f"Created TimeSeries with {len(series)} points")
    logger.info(f"Series range: {series.start_time()} to {series.end_time()}")
    
    # Split into train and validation
    split_point = int(len(series) * (1 - val_ratio))
    
    train_series = series[:split_point]
    val_series = series[split_point:]
    
    logger.info(f"Train series: {len(train_series)} points ({train_series.start_time()} to {train_series.end_time()})")
    logger.info(f"Validation series: {len(val_series)} points ({val_series.start_time()} to {val_series.end_time()})")
    
    return train_series, val_series