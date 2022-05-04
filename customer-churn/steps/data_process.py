import pandas as pd
from zenml.logger import get_logger
from zenml.steps import Output, step

from .src.data_processing import DataProcessor

logger = get_logger(__name__)


@step
def encode_cat_cols(data: pd.DataFrame) -> Output(output_data=pd.DataFrame):
    """Encode categorical columns.

    Args:
        data: pd.DataFrame
    """
    try:
        data_processor = DataProcessor()
        data = data_processor.encode_categorical_columns(data)
        return data
    except ValueError:
        logger.error(
            "Categorical columns encoding failed due to not matching the type of the input data. Recheck the type of your input data."
        )
        raise ValueError


@step
def mean_encoding(data: pd.DataFrame) -> Output(output_data=pd.DataFrame):
    """Mean encoding of categorical columns.

    Args:
        data: pd.DataFrame
    """
    try:
        data_processor = DataProcessor()
        data = data_processor.mean_encoding(data)
        return data
    except ValueError:
        logger.error("Mean encoding failed, try rechecking the type of your input data.")
        raise ValueError
    except Exception as e:
        logger.error(e)
        raise e


@step
def handle_imbalanced_data(data: pd.DataFrame) -> Output(output_data=pd.DataFrame):
    """Handle imbalanced data.

    Args:
        data: pd.DataFrame
    """
    try:
        data_processor = DataProcessor()
        data = data_processor.handle_imbalanced_data(data)
        return data
    except ValueError:
        logger.error(
            "Handle imbalanced data failed, try rechecking the type of your input data and ensure that the type of your input data is a Dataframe."
        )
        raise ValueError
    except Exception as e:
        logger.error(e)
        raise e


@step
def drop_cols(data: pd.DataFrame) -> Output(output_data=pd.DataFrame):
    """Drop columns.

    Args:
        data: pd.DataFrame
    """
    try:
        data_processor = DataProcessor()
        data = data_processor.drop_columns(data)
        return data
    except ValueError:
        logger.error(
            "Drop columns failed due to not matching the type of the input data, Recheck the type of your input data."
        )
        raise ValueError
    except Exception as e:
        logger.error(e)
        raise e
