import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def data_splitter(
    data: pd.DataFrame,
) -> Output(X_train=pd.DataFrame, X_test=pd.DataFrame, y_train=pd.DataFrame, y_test=pd.DataFrame):
    """Data Splitter step which splits the data into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        X = data.drop(["Churn"], axis=1)
        y = data["Churn"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
