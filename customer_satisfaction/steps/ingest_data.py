from zenml.steps import step
import pandas as pd
import logging 

class IngestData: 
    '''
    Data ingestion class which ingests data from the source and returns a DataFrame. 

    '''
    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame: 
        df = pd.read_csv("./data/olist_customers_dataset.csv") 
        return df 

@step
def ingest_data() -> pd.DataFrame:
    """ 
    Args:  
        None 
    Returns: 
        df: pd.DataFrame
    """ 
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e

