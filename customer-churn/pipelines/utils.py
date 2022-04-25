import pandas as pd
from steps.src.data_processing import DataProcessor
from zenml.logger import get_logger

logger = get_logger(__name__)


def get_data_for_test():
    try:
        df = pd.read_csv("./data/customer-churn-data.csv")
        df = df.sample(n=100)
        data_clean = DataProcessor()
        df = data_clean.encode_categorical_columns(df)
        df.drop(["Churn"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logger.error(e)
        raise e
