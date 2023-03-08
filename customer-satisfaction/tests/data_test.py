import logging

import pytest
from model.data_cleaning import DataCleaning
from steps.ingest_data import IngestData
from zenml.steps import step


@step
def data_test_prep_step():
    """Test the shape of the data after the data cleaning step."""
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        data_cleaning = DataCleaning(df)
        df = data_cleaning.preprocess_data()
        X_train, X_test, y_train, y_test = data_cleaning.divide_data(df)

        assert X_train.shape == (
            92487,
            12,
        ), "The shape of the training set is not correct."
        assert y_train.shape == (
            92487,
        ), "The shape of labels of training set is not correct."
        assert X_test.shape == (
            23122,
            12,
        ), "The shape of the testing set is not correct."
        assert y_test.shape == (
            23122,
        ), "The shape of labels of testing set is not correct."
        logging.info("Data Shape Assertion test passed.")
    except Exception as e:
        pytest.fail(e)


@step
def check_data_leakage(X_train, X_test):
    """Test if there is any data leakage."""
    try:
        assert (
            len(X_train.index.intersection(X_test.index)) == 0
        ), "There is data leakage."
        logging.info("Data Leakage test passed.")
    except Exception as e:
        pytest.fail(e)


@step
def test_ouput_range_features(df):
    """Test output range of the target variable between 0 - 5"""
    try:
        assert (
            df["review_score"].max() <= 5
        ), "The output range of the target variable is not correct."
        assert (
            df["review_score"].min() >= 0
        ), "The output range of the target variable is not correct."
        logging.info("Output Range Assertion test passed.")
    except Exception as e:
        pytest.fail(e)
