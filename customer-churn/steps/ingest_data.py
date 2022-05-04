import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


class DataIngestion:
    """Class for data ingestion"""

    def __init__(self) -> None:
        self.path = "data"

    def read_data(self) -> pd.DataFrame:
        """Reads data from csv files and merges the csv files into a single DataFrame"""
        try:
            customer_churn_df = pd.read_csv(os.path.join(self.path, "customer-churn-data.csv"))
            return customer_churn_df
        except Exception as e:
            logging.error(e)


@step
def ingest_data() -> Output(data=pd.DataFrame):
    """Data ingestion step which takes data from the source and returns a DataFrame.

    Args:
        data: pd.DataFrame
    """
    try:
        data_ingestion = DataIngestion()
        customer_churn_df = data_ingestion.read_data()
        return customer_churn_df
    except Exception as e:
        logging.error(e)
        raise e
