from typing import Dict, List, Tuple

import pandas as pd
from zenml import step


@step
def preprocess_data(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str]]:
    """
    Prepare data for forecasting with Prophet.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        test_size: Proportion of data to use for testing

    Returns:
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers (store-item combinations)
    """
    print(f"Preprocessing sales data with shape: {sales_data.shape}")

    # Convert date to datetime
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create unique series ID for each store-item combination
    sales_data["series_id"] = sales_data["store"] + "-" + sales_data["item"]

    # Get list of unique series
    series_ids = sales_data["series_id"].unique().tolist()
    print(f"Found {len(series_ids)} unique store-item combinations")

    # Create Prophet-formatted dataframes (ds, y) for each series
    train_data_dict = {}
    test_data_dict = {}

    for series_id in series_ids:
        # Filter data for this series
        series_data = sales_data[sales_data["series_id"] == series_id].copy()
        series_data = series_data.sort_values("date")

        # Rename columns for Prophet
        prophet_data = series_data[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        # Split into train and test
        cutoff_idx = int(len(prophet_data) * (1 - test_size))
        train_data = prophet_data.iloc[:cutoff_idx].copy()
        test_data = prophet_data.iloc[cutoff_idx:].copy()

        # Store in dictionaries
        train_data_dict[series_id] = train_data
        test_data_dict[series_id] = test_data

    print(
        f"Data preprocessing complete. Train periods: {len(train_data)}, Test periods: {len(test_data)}"
    )

    return train_data_dict, test_data_dict, series_ids
