from google.oauth2 import service_account
from zenml.steps import step, BaseStepConfig
import pandas_gbq
import pandas as pd

class BigQueryImporterConfig(BaseStepConfig):
    """Config class for Google BigQuery.
    
    Attributes:
        query: SQL query to specify what data should be fetched.
        project_id: GCP Project ID that contains the data.
    """

    query: str = 'SELECT * FROM `computas_dataset.wind_forecast`'
    project_id: str = 'computas-project-345810'

@step
def bigquery_importer(config: BigQueryImporterConfig) -> pd.DataFrame:
    """Import el. power and wind forecast data from BQ.

    Args:
        config: SQL query to get the data and GCP Project ID. 

    Returns:
        pd.DataFrame 
    """
    credentials = service_account.Credentials.from_service_account_file('credentials.json')
    return pandas_gbq.read_gbq(config.query, project_id = config.project_id, credentials = credentials)

