import logging
import os

import numpy as np
import pandas as pd
from zenml import step

logger = logging.getLogger(__name__)


@step
def load_data() -> pd.DataFrame:
    """Load synthetic retail sales data for forecasting."""
    data_dir = os.path.join(os.getcwd(), "data")
    sales_path = os.path.join(data_dir, "sales.csv")

    if os.path.exists(sales_path):
        # Load real data if available
        sales_df = pd.read_csv(sales_path)
        logger.info(f"Loaded {len(sales_df)} sales records from file.")
    else:
        logger.info("Generating synthetic retail sales data...")
        # Create synthetic dataset with retail patterns
        np.random.seed(42)  # For reproducibility

        # Generate date range for 3 months
        date_range = pd.date_range("2024-01-01", periods=90, freq="D")

        # Create stores and items
        stores = ["Store_1", "Store_2"]
        items = ["Item_A", "Item_B", "Item_C"]

        records = []
        for date in date_range:
            # Calendar features
            is_weekend = 1 if date.dayofweek >= 5 else 0
            is_holiday = 1 if date.day == 1 or date.day == 15 else 0
            is_promo = 1 if 10 <= date.day <= 20 else 0

            for store in stores:
                for item in items:
                    # Base demand with factors
                    base_demand = 100
                    store_factor = 1.5 if store == "Store_1" else 0.8
                    item_factor = (
                        1.2
                        if item == "Item_A"
                        else 1.0
                        if item == "Item_B"
                        else 0.7
                    )
                    weekday_factor = 1.5 if is_weekend else 1.0
                    holiday_factor = 2.0 if is_holiday else 1.0
                    promo_factor = 1.8 if is_promo else 1.0

                    # Add random noise
                    noise = np.random.normal(1, 0.1)

                    # Calculate final sales
                    sales = int(
                        base_demand
                        * store_factor
                        * item_factor
                        * weekday_factor
                        * holiday_factor
                        * promo_factor
                        * noise
                    )
                    sales = max(0, sales)

                    records.append(
                        {
                            "date": date,
                            "store": store,
                            "item": item,
                            "sales": sales,
                        }
                    )

        # Create DataFrame
        sales_df = pd.DataFrame(records)

        # Save synthetic data
        os.makedirs(data_dir, exist_ok=True)
        sales_df.to_csv(sales_path, index=False)

    return sales_df
