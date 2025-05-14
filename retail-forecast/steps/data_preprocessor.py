from zenml import step
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from typing_extensions import Annotated

@step
def preprocess_data(sales_data: pd.DataFrame, calendar_data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "train_data"],
    Annotated[pd.DataFrame, "val_data"], 
    Annotated[pd.DataFrame, "test_data"]
]:
    """
    Preprocess data for retail forecasting, creating:
    - Time series features (day of week, month, etc.)
    - Lagged features and rolling statistics
    - Store and item embeddings
    - Train/validation/test splits based on time
    
    Args:
        sales_data: Raw sales data with store/item/date/sales columns
        calendar_data: Calendar data with date/event information
        
    Returns:
        train_data: Processed training data
        val_data: Processed validation data
        test_data: Processed test data
    """
    sales_df = sales_data
    calendar_df = calendar_data

    # Ensure date columns are datetime
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])

    # Merge calendar features with sales data
    df = pd.merge(sales_df, calendar_df, on="date", how="left")

    # Create time-based features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week_of_year"] = df["date"].dt.isocalendar().week

    # Create unique identifier for each time series (store-item combination)
    df["series_id"] = df["store"] + "_" + df["item"]

    # Create a date column for sorting
    df = df.sort_values(["series_id", "date"])

    # Create lag features for each series
    for lag in [1, 7, 14]:  # 1 day, 1 week, 2 weeks
        df[f"sales_lag_{lag}"] = df.groupby("series_id")["sales"].shift(lag)

    # Create rolling window features
    for window in [7, 14, 28]:  # 1 week, 2 weeks, 4 weeks
        # Rolling mean
        df[f"sales_rolling_mean_{window}"] = df.groupby("series_id")[
            "sales"
        ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        # Rolling std
        df[f"sales_rolling_std_{window}"] = df.groupby("series_id")[
            "sales"
        ].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
        )

    # Prepare categorical features for PyTorch Forecasting
    # For PyTorch Forecasting, categorical variables must be strings
    categorical_feats = ["store", "item", "series_id", "day_of_week", "month"]

    for feat in categorical_feats:
        # Create mapping
        mapping = {val: f"cat_{idx}" for idx, val in enumerate(df[feat].unique())}
        df[f"{feat}_encoded"] = df[feat].map(mapping).astype(str)

    # Fill any missing values in the engineered features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Split data into train, validation, test sets by time
    # But ensure validation and test have some overlap with training
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=28)
    val_start = test_start - pd.Timedelta(days=14)
    
    # Overlap: Include the last 7 days of training in validation, and last 7 days of validation in test
    train_end = val_start + pd.Timedelta(days=7)  # Overlap of 7 days
    val_end = test_start + pd.Timedelta(days=7)   # Overlap of 7 days
    
    train_df = df[df["date"] <= train_end].copy()
    val_df = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
    test_df = df[df["date"] >= test_start].copy()
    
    print(
        f"Train set: {len(train_df)} records, {train_df['date'].min()} to {train_df['date'].max()}"
    )
    print(
        f"Validation set: {len(val_df)} records, {val_df['date'].min()} to {val_df['date'].max()}"
    )
    print(
        f"Test set: {len(test_df)} records, {test_df['date'].min()} to {test_df['date'].max()}"
    )

    # Normalize numeric features for neural networks
    scaler = StandardScaler()

    # Select features to scale (excluding target and IDs)
    features_to_scale = [
        "price",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_rolling_mean_7",
        "sales_rolling_mean_14",
        "sales_rolling_mean_28",
        "sales_rolling_std_7",
        "sales_rolling_std_14",
        "sales_rolling_std_28",
    ]

    # Fit scaler on training data only to avoid data leakage
    for feat in features_to_scale:
        if (
            feat in train_df.columns
        ):  # Some features might not exist if we have insufficient history
            scaler_fit = scaler.fit(train_df[[feat]])
            train_df[f"{feat}_scaled"] = scaler.transform(train_df[[feat]])
            val_df[f"{feat}_scaled"] = scaler.transform(val_df[[feat]])
            test_df[f"{feat}_scaled"] = scaler.transform(test_df[[feat]])

    # Prepare data for PyTorch Forecasting format
    # Create a continuous time index across all datasets
    # Sort all dates and create a single mapping
    all_dates = sorted(df["date"].unique())
    time_idx_mapping = {date: idx for idx, date in enumerate(all_dates)}
    
    # Apply the mapping to all datasets
    train_df["time_idx"] = train_df["date"].map(time_idx_mapping)
    val_df["time_idx"] = val_df["date"].map(time_idx_mapping)
    test_df["time_idx"] = test_df["date"].map(time_idx_mapping)

    # Convert time features to string categories as well
    for df_split in [train_df, val_df, test_df]:
        df_split["day_of_week"] = df_split["day_of_week"].astype(str)
        df_split["month"] = df_split["month"].astype(str)

    # Return just the dataframes directly
    return train_df, val_df, test_df
