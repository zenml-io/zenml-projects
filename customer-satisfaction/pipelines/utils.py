import logging

import pandas as pd
from zenml import ModelVersion
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from model.data_cleaning import DataCleaning


def get_data_for_test():
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)
        data_clean = DataCleaning(df)
        df = data_clean.preprocess_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e


docker_settings = DockerSettings(required_integrations=[MLFLOW])
model_version = ModelVersion(
    name="Customer_Satisfaction_Predictor",
    description="Predictor of Customer Satisfaction.",
    delete_new_version_on_failure=True,
    tags=["classification", "customer_satisfaction"],
)
