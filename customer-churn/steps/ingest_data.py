import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from whylogs import DatasetProfile  # type: ignore
from zenml.integrations.whylogs.whylogs_context import WhylogsContext
from zenml.steps import Output, step
from zenml.steps.step_context import StepContext


class DataIngestion:
    """Class for data Ingestion"""

    def __init__(self) -> None:
        self.path = "customer-churn/data"

    def read_data(self) -> pd.DataFrame:
        """Reads data from csv files and merge the csv files into one dataframe"""
        try:
            customer_churn_df = pd.read_csv(os.path.join(self.path, "customer-churn-data.csv"))
            return customer_churn_df
        except Exception as e:
            logging.error(e)


@step
def ingest_data() -> Output(data=pd.DataFrame):
    """Data Ingestion step which ingests data from the source and returns a DataFrame.

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
