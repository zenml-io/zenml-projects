from zenml import step
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler


@step
def preprocess_data(data: dict) -> dict:
    """
    Preprocess data for retail forecasting, creating:
    - Time series features (day of week, month, etc.)
    - Lagged features and rolling statistics
    - Store and item embeddings
    - Train/validation/test splits based on time
    """
    sales_df = data["sales"]
    calendar_df = data["calendar"]

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
    # First, we need to ensure all categorical features start from 0 and are continuous
    categorical_feats = ["store", "item", "series_id", "day_of_week", "month"]

    for feat in categorical_feats:
        # Create mapping
        mapping = {val: idx for idx, val in enumerate(df[feat].unique())}
        df[f"{feat}_encoded"] = df[feat].map(mapping)

    # Fill any missing values in the engineered features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Split data into train, validation, test sets by time
    # Last 28 days for test, previous 14 days for validation
    max_date = df["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=28)
    val_cutoff = test_cutoff - pd.Timedelta(days=14)

    train_df = df[df["date"] <= val_cutoff].copy()
    val_df = df[(df["date"] > val_cutoff) & (df["date"] <= test_cutoff)].copy()
    test_df = df[df["date"] > test_cutoff].copy()

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
    # Group by time-series and prepare for modeling
    time_idx_mapping = {
        date: idx for idx, date in enumerate(sorted(df["date"].unique()))
    }

    train_df["time_idx"] = train_df["date"].map(time_idx_mapping)
    val_df["time_idx"] = val_df["date"].map(time_idx_mapping)
    test_df["time_idx"] = test_df["date"].map(time_idx_mapping)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "time_idx_mapping": time_idx_mapping,
        "scaler": scaler,
        "features": {
            "categorical": [f"{feat}_encoded" for feat in categorical_feats],
            "continuous": [
                f"{feat}_scaled"
                for feat in features_to_scale
                if f"{feat}_scaled" in train_df.columns
            ],
            "time_varying": [
                "day_of_week",
                "day_of_month",
                "month",
                "is_weekend",
                "is_holiday",
                "is_promo",
            ]
            + [
                f"{feat}_scaled"
                for feat in features_to_scale
                if f"{feat}_scaled" in train_df.columns
            ],
            "static": ["store_encoded", "item_encoded"],
        },
    }
