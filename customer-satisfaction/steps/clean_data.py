import logging
from typing import Annotated, Tuple

import pandas as pd
from model.data_cleaning import DataCleaning
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
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
