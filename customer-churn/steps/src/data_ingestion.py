import logging
import os

import pandas as pd


class DataIngestion:
    """Class for data Ingestion"""

    def __init__(self) -> None:
        """Initialize the DataIngestion class."""
        self.path = "data"

    def read_data(self) -> pd.DataFrame:
        """Reads data from csv files and merge the csv files into one dataframe"""
        try:
            customer_churn_df = pd.read_csv(os.path.join(self.path, "customer-churn-data.csv"))
            return customer_churn_df
        except Exception as e:
            logging.error(e)
