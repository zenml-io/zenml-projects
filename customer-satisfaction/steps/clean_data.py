import logging

import pandas as pd
from model.data_cleaning import DataCleaning
from zenml.steps import Output, step


@step
def clean_data(
    data: pd.DataFrame,
) -> Output(
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        data_cleaning = DataCleaning(data)
        df = data_cleaning.preprocess_data()
        x_train, x_test, y_train, y_test = data_cleaning.divide_data(df)
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
