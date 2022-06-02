from google.oauth2 import service_account
from zenml.steps import step, BaseStepConfig
import pandas_gbq
import pandas as pd

class BigQueryImporterConfig(BaseStepConfig):
    query: str = 'SELECT * FROM `computas_dataset.wind_forecast`'
    project_id: str = 'computas-project-345810'

@step
def bigquery_importer(config: BigQueryImporterConfig) -> pd.DataFrame:
    credentials = service_account.Credentials.from_service_account_file('credentials.json')
    return pandas_gbq.read_gbq(config.query, project_id = config.project_id, credentials = credentials)

