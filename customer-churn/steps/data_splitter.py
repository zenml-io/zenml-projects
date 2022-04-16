import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml.steps import Output, step


@step
def data_splitter(
    data: pd.DataFrame,
) -> Output(train=pd.DataFrame, test=pd.DataFrame):
    """Data Splitter step which splits the data into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        return train, test
    except Exception as e:
        logging.error(e)
        raise e
