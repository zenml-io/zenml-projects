from zenml.steps import step, Output
from model.data_ingestion import IngestData
import pandas as pd


@step
def ingest_data() -> pd.DataFrame:
    """ 
    Args:  
        None 
    Returns: 
        df: pd.DataFrame
    """
    ingest_data = IngestData()
    df = ingest_data.get_data()
    return df

