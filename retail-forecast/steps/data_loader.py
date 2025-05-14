from zenml import step
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from typing_extensions import Annotated

@step
def load_data() -> Tuple[Annotated[pd.DataFrame, "sales_data"], Annotated[pd.DataFrame, "calendar_data"]]:
    """Load the dataset. Looks for CSVs in ./data; falls back to synthetic demo data."""
    data_dir = os.path.join(os.getcwd(), "data")
    sales_path = os.path.join(data_dir, "sales.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")

    if os.path.exists(sales_path) and os.path.exists(calendar_path):
        # Load real data if available
        sales_df = pd.read_csv(sales_path)
        calendar_df = pd.read_csv(calendar_path)
        print(f"Loaded {len(sales_df)} sales records from file.")
    else:
        print("Using synthetic data as no real data files found.")
        # Create a larger synthetic dataset with retail patterns
        np.random.seed(42)  # For reproducibility

        # Generate date range for 3 months
        date_range = pd.date_range("2024-01-01", periods=90, freq="D")

        # Create multiple stores and items
        stores = ["Store_1", "Store_2", "Store_3"]
        items = ["Item_A", "Item_B", "Item_C", "Item_D", "Item_E"]

        # Generate sales data with patterns:
        # - Weekly seasonality (weekends have higher sales)
        # - Different baseline per item and store
        # - Some trend components
        # - Special events (like promotions)

        records = []
        # Create calendar data with weekday, events, etc.
        calendar_data = []

        for date in date_range:
            # Calendar features
            is_weekend = 1 if date.dayofweek >= 5 else 0
            month = date.month
            day = date.day

            # Mark some days as events/holidays (e.g., every 15th day)
            is_holiday = 1 if day == 1 or day == 15 else 0

            # Mark some periods as promotion periods
            is_promo = 1 if 10 <= day <= 20 else 0

            calendar_data.append(
                {
                    "date": date,
                    "weekday": date.dayofweek,
                    "month": month,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "is_promo": is_promo,
                }
            )

            for store in stores:
                # Store-specific baseline (some stores sell more)
                store_factor = (
                    1.5
                    if store == "Store_1"
                    else 1.0
                    if store == "Store_2"
                    else 0.8
                )

                for item in items:
                    # Item-specific baseline (some items sell more)
                    item_factor = (
                        1.2
                        if item == "Item_A"
                        else 1.0
                        if item in ["Item_B", "Item_C"]
                        else 0.7
                    )

                    # Base demand with weekly pattern
                    base_demand = 10 * store_factor * item_factor
                    weekday_factor = 1.5 if is_weekend else 1.0

                    # Add holiday effect
                    holiday_factor = 2.0 if is_holiday else 1.0

                    # Add promotion effect
                    promo_factor = 1.8 if is_promo else 1.0

                    # Add slight trend (increase over time)
                    trend = 1.0 + (date_range.get_loc(date) * 0.003)

                    # Add seasonality (e.g., monthly effect)
                    monthly_seasonality = 1.0 + 0.2 * np.sin(
                        2 * np.pi * date.day / 30
                    )

                    # Add random noise
                    noise = np.random.normal(1, 0.2)

                    # Calculate final sales
                    sales = int(
                        base_demand
                        * weekday_factor
                        * holiday_factor
                        * promo_factor
                        * trend
                        * monthly_seasonality
                        * noise
                    )

                    # Ensure no negative sales
                    sales = max(0, sales)

                    records.append(
                        {
                            "date": date,
                            "store": store,
                            "item": item,
                            "sales": sales,
                            # We're using price as a feature too
                            "price": round(
                                10 * item_factor * (0.9 if is_promo else 1.0),
                                2,
                            ),
                        }
                    )

        # Create DataFrames
        sales_df = pd.DataFrame(records)
        calendar_df = pd.DataFrame(calendar_data)

        # Optionally save synthetic data for future use
        os.makedirs(data_dir, exist_ok=True)
        sales_df.to_csv(os.path.join(data_dir, "sales.csv"), index=False)
        calendar_df.to_csv(os.path.join(data_dir, "calendar.csv"), index=False)

    # Return both dataframes as a dictionary
    return sales_df, calendar_df
