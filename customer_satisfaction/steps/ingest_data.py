from zenml.steps import step
from model.data_cleaning import DataCleaning
import pandas as pd

class IngestData: 
    '''
    Data ingestion class which ingests data from the source and returns a DataFrame. 

    '''
    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame: 
        df = pd.read_csv("./data/olist_customers_dataset.csv") 
        return df 

    def get_data_for_test():
        df = pd.read_csv("data/olist_customers_dataset.csv")
        df = df.sample(n=100) 
        data_clean = DataCleaning(df) 
        df = data_clean.preprocess_data() 
        df.drop(["review_score"], axis=1, inplace=True)    
        result = df.to_json(orient="split")
        return result 

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

