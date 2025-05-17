import logging
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


@step
def validate_data(
    sales_data: pd.DataFrame, calendar_data: pd.DataFrame
) -> Tuple[
    Annotated[pd.DataFrame, "sales_data_validated"],
    Annotated[pd.DataFrame, "calendar_data_validated"],
]:
    """Validate retail sales data, checking for common issues like:
    - Missing values
    - Negative sales
    - Duplicate records
    - Date continuity
    - Extreme outliers
    """
    sales_df = sales_data
    calendar_df = calendar_data

    # Check for missing values in critical fields
    for df_name, df in [("Sales", sales_df), ("Calendar", calendar_df)]:
        if df.isnull().any().any():
            missing_cols = df.columns[df.isnull().any()].tolist()
            logger.info(
                f"Warning: {df_name} data contains missing values in columns: {missing_cols}"
            )
            # Fill missing values appropriately based on column type
            for col in missing_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # For numeric columns, fill with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # For categorical/text columns, fill with most common value
                    df[col] = df[col].fillna(
                        df[col].mode()[0]
                        if not df[col].mode().empty
                        else "UNKNOWN"
                    )

    # Check for and fix negative sales (a common data quality issue in retail)
    neg_sales = (sales_df["sales"] < 0).sum()
    if neg_sales > 0:
        logger.info(
            f"Warning: Found {neg_sales} records with negative sales. Setting to zero."
        )
        sales_df.loc[sales_df["sales"] < 0, "sales"] = 0

    # Check for duplicate records
    duplicates = sales_df.duplicated(subset=["date", "store", "item"]).sum()
    if duplicates > 0:
        logger.info(
            f"Warning: Found {duplicates} duplicate store-item-date records. Keeping the last one."
        )
        sales_df = sales_df.drop_duplicates(
            subset=["date", "store", "item"], keep="last"
        )

    # Check for date continuity in calendar
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    date_diff = calendar_df["date"].diff().dropna()
    if not (date_diff == pd.Timedelta(days=1)).all():
        logger.info(
            "Warning: Calendar dates are not continuous. Some days may be missing."
        )

    # Detect extreme outliers (values > 3 std from mean within each item-store combination)
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    outlier_count = 0

    # Group by store and item to identify outliers within each time series
    for (store, item), group in sales_df.groupby(["store", "item"]):
        mean_sales = group["sales"].mean()
        std_sales = group["sales"].std()

        if std_sales > 0:  # Avoid division by zero
            # Calculate z-score
            z_scores = (group["sales"] - mean_sales) / std_sales

            # Flag extreme outliers (|z| > 3)
            outlier_mask = abs(z_scores) > 3
            outlier_count += outlier_mask.sum()

            # Cap outliers (winsorize) rather than removing them
            if outlier_mask.any():
                cap_upper = mean_sales + 3 * std_sales
                sales_df.loc[
                    group[outlier_mask & (group["sales"] > cap_upper)].index,
                    "sales",
                ] = cap_upper

    if outlier_count > 0:
        logger.info(
            f"Warning: Detected and capped {outlier_count} extreme sales outliers."
        )

    # Ensure all dates in sales exist in calendar
    if not set(sales_df["date"].dt.date).issubset(
        set(calendar_df["date"].dt.date)
    ):
        logger.info(
            "Warning: Some sales dates don't exist in the calendar data."
        )

    return sales_df, calendar_df
