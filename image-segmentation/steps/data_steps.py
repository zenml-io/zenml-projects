import pandas as pd
from src.data.prepare_data import PrepareDataFrame
from zenml.logger import get_logger
from zenml.steps import Output, step

logger = get_logger(__name__)


@step
def prepare_data() -> Output(processed_dataframe=pd.DataFrame):
    """
    TODO:
    """
    try:
        prep_df = PrepareDataFrame("./data/archive/train.csv")
        processed_dataframe = prep_df.prepare_data()
        return processed_dataframe
    except Exception as e:
        logger.error(e)
        raise e
