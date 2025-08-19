"""
Data ingestion step for FloraCast.
"""

import os
from typing import Optional, Annotated
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


def generate_ecommerce_data(num_days: int = 730) -> pd.DataFrame:
    """
    Generate synthetic ecommerce daily sales data with realistic patterns.

    Args:
        num_days: Number of days to generate

    Returns:
        DataFrame with columns 'ds' (date) and 'y' (sales)
    """
    logger.info(f"Generating synthetic ecommerce data for {num_days} days")

    # Create date range
    start_date = datetime.now() - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")

    # Base trend with multiple components
    base_trend = np.linspace(100, 200, num_days)

    # Add yearly seasonality (higher in winter/holidays)
    yearly_cycle = 20 * np.sin(
        2 * np.pi * np.arange(num_days) / 365.25 - np.pi / 2
    )

    # Weekly seasonality (higher sales on weekends)
    weekly_pattern = np.array(
        [0.8, 0.85, 0.9, 0.95, 1.0, 1.3, 1.2]
    )  # Mon to Sun
    weekly_seasonality = np.tile(weekly_pattern, (num_days // 7) + 1)[
        :num_days
    ]

    # Monthly cycle (end of month sales bumps)
    days_in_month = pd.date_range(
        start=start_date, periods=num_days, freq="D"
    ).day
    monthly_boost = np.where(
        (days_in_month >= 28) | (days_in_month <= 3), 1.2, 1.0
    )

    # Add some autocorrelated noise for realism
    np.random.seed(42)
    noise = np.random.normal(0, 8, num_days)
    # Add some persistence to noise
    for i in range(1, num_days):
        noise[i] = 0.3 * noise[i - 1] + 0.7 * noise[i]

    # Combine all components
    sales = (
        base_trend + yearly_cycle
    ) * weekly_seasonality * monthly_boost + noise
    sales = np.maximum(sales, 20)  # Ensure positive values
    sales = sales.astype(int)

    return pd.DataFrame({"ds": dates, "y": sales})


@step
def ingest_data(
    data_source: str = "ecommerce_default",
    data_path: Optional[str] = None,
    datetime_col: str = "ds",
    target_col: str = "y",
    infer: bool = False,
) -> Annotated[pd.DataFrame, "raw_data"]:
    """
    Ingest data based on configuration parameters.

    Args:
        data_source: Type of data source ("ecommerce_default" or "csv")
        data_path: Path to CSV file (when data_source is "csv")
        datetime_col: Name of datetime column
        target_col: Name of target column
        infer: If True, simulate real-time data ingestion for inference

    Returns:
        DataFrame with datetime and target columns
    """
    if data_source == "ecommerce_default":
        # Generate or load default ecommerce data
        csv_file_path = "data/ecommerce_daily.csv"

        if not os.path.exists(csv_file_path):
            logger.info(
                "Default ecommerce data not found, generating new data"
            )
            os.makedirs("data", exist_ok=True)
            df = generate_ecommerce_data()
            df.to_csv(csv_file_path, index=False)
            logger.info(
                f"Generated and saved ecommerce data to {csv_file_path}"
            )
        else:
            if infer:
                logger.info(
                    f"🔄 INFERENCE MODE: Simulating real-time data ingestion from {csv_file_path}"
                )
                logger.info(
                    "📡 In production, this would connect to real-time data sources like:"
                )
                logger.info("   - Database queries for latest sales data")
                logger.info("   - API calls to fetch recent transactions")
                logger.info("   - Stream processing from Kafka/Kinesis")
                logger.info("   - Data lake queries for updated metrics")
                logger.info(
                    "📊 For demo purposes, loading same data as training to show inference workflow"
                )
            else:
                logger.info(
                    f"Loading existing ecommerce data from {csv_file_path}"
                )
            df = pd.read_csv(csv_file_path, parse_dates=["ds"])

    elif data_source == "csv":
        # Load from specified CSV path
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV file not found: {data_path}")

        logger.info(f"Loading data from CSV: {data_path}")
        df = pd.read_csv(data_path)

        # Parse datetime column
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        else:
            raise ValueError(
                f"Datetime column '{datetime_col}' not found in CSV"
            )

    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # Validate required columns
    if datetime_col not in df.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in data")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    # Ensure proper data types
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df[target_col] = pd.to_numeric(df[target_col])

    # Sort by datetime
    df = df.sort_values(datetime_col).reset_index(drop=True)

    logger.info(f"Ingested {len(df)} rows of data")
    logger.info(
        f"Date range: {df[datetime_col].min()} to {df[datetime_col].max()}"
    )
    logger.info(
        f"Target stats: mean={df[target_col].mean():.2f}, std={df[target_col].std():.2f}"
    )

    return df
