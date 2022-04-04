from zenml.steps import step, Output
from model.data_cleaning import DataCleaning
import pandas as pd


@step
def clean_data(
    data: pd.DataFrame,
) -> Output(
    x_train=pd.DataFrame,
    x_test=pd.DataFrame,
    y_train=pd.Series,
    y_test=pd.Series,
):
    """ 
    Data Cleaning class which preprocesses the data and divides it into train and test data. 

    Args:
        data: pd.DataFrame 
    """
    data_cleaning = DataCleaning(data)
    df = data_cleaning.preprocess_data()
    x_train, x_test, y_train, y_test = data_cleaning.divide_data(df) 
    print(x_train.head())
    return x_train, x_test, y_train, y_test
