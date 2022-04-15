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
        self.path = "data"

    def read_data(self) -> pd.DataFrame:
        """Reads data from csv files and merge the csv files into one dataframe"""
        try:
            customer_churn_df = pd.read_csv(os.path.join(self.path, "customer-churn-data.csv"))
            return customer_churn_df
        except Exception as e:
            logging.error(e)


@step
def ingest_data(
    context: StepContext,
) -> Output(data=pd.DataFrame):
    """Data Ingestion step which ingests data from the source and returns a DataFrame.

    Args:
        data: pd.DataFrame
    """
    try:
        data_ingestion = DataIngestion()
        customer_churn_df = data_ingestion.read_data()
        whylogs_context = WhylogsContext(context)
        profile = whylogs_context.profile_dataframe(
            customer_churn_df, dataset_name="input_data", tags={"datasetId": "customer-churn-1"}
        )
        return customer_churn_df
    except Exception as e:
        logging.error(e)
        raise e


@step
def data_splitter(
    context: StepContext,
    data: pd.DataFrame,
) -> Output(
    train_data=pd.DataFrame,
    test_data=pd.DataFrame,
    train_data_profile=DatasetProfile,
    test_data_profile=DatasetProfile,
):
    """Data Splitter step which splits the data into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        whylogs_context = WhylogsContext(context)
        train_data_profile = whylogs_context.profile_dataframe(
            train, dataset_name="train_data", tags={"datasetId": "customer-churn-2"}
        )
        test_data_profile = whylogs_context.profile_dataframe(
            test, dataset_name="test_data", tags={"datasetId": "customer-churn-3"}
        )
        return train, test, train_data_profile, test_data_profile
    except Exception as e:
        logging.error(e)
        raise e
