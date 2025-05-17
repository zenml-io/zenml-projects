import logging
from typing import Dict, List, Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


@step
def preprocess_data(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "training_data"],
    Annotated[Dict[str, pd.DataFrame], "testing_data"],
    Annotated[List[str], "series_identifiers"],
]:
    """Prepare data for forecasting with Prophet.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        test_size: Proportion of data to use for testing

    Returns:
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers (store-item combinations)
    """
    logger.info(f"Preprocessing sales data with shape: {sales_data.shape}")

    # Convert date to datetime
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create unique series ID for each store-item combination
    sales_data["series_id"] = sales_data["store"] + "-" + sales_data["item"]

    # Get list of unique series
    series_ids = sales_data["series_id"].unique().tolist()
    logger.info(f"Found {len(series_ids)} unique store-item combinations")

    # Create Prophet-formatted dataframes (ds, y) for each series
    train_data_dict = {}
    test_data_dict = {}

    for series_id in series_ids:
        # Filter data for this series
        series_data = sales_data[sales_data["series_id"] == series_id].copy()

        # Sort by date and drop any duplicates
        series_data = series_data.sort_values("date").drop_duplicates(
            subset=["date"]
        )

        # Rename columns for Prophet
        prophet_data = series_data[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        # Ensure no NaN values
        prophet_data = prophet_data.dropna()

        if len(prophet_data) < 2:
            logger.info(
                f"WARNING: Not enough data for series {series_id}, skipping"
            )
            continue

        # Make sure we have at least one point in test set
        min_test_size = max(1, int(len(prophet_data) * test_size))

        if len(prophet_data) <= min_test_size:
            # If we don't have enough data, use half for training and half for testing
            cutoff_idx = len(prophet_data) // 2
        else:
            cutoff_idx = len(prophet_data) - min_test_size

        # Split into train and test
        train_data = prophet_data.iloc[:cutoff_idx].copy()
        test_data = prophet_data.iloc[cutoff_idx:].copy()

        # Ensure we have data in both splits
        if len(train_data) == 0 or len(test_data) == 0:
            logger.info(
                f"WARNING: Empty split for series {series_id}, skipping"
            )
            continue

        # Store in dictionaries
        train_data_dict[series_id] = train_data
        test_data_dict[series_id] = test_data

        logger.info(
            f"Series {series_id}: {len(train_data)} train points, {len(test_data)} test points"
        )

    if not train_data_dict:
        raise ValueError("No valid series data after preprocessing!")

    # Get a sample series to print details
    sample_id = next(iter(train_data_dict))
    sample_train = train_data_dict[sample_id]
    sample_test = test_data_dict[sample_id]

    logger.info(f"Sample series {sample_id}:")
    logger.info(f"  Train data shape: {sample_train.shape}")
    logger.info(
        f"  Train date range: {sample_train['ds'].min()} to {sample_train['ds'].max()}"
    )
    logger.info(f"  Test data shape: {sample_test.shape}")
    logger.info(
        f"  Test date range: {sample_test['ds'].min()} to {sample_test['ds'].max()}"
    )

    return train_data_dict, test_data_dict, series_ids
