# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import pandas as pd
from google.cloud import bigquery
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_data_bq(
    df: pd.DataFrame,
    table_id: str = "your-project.your_dataset.ecb_raw_data",
    write_disposition: str = "WRITE_TRUNCATE",
) -> str:
    """Load data to BigQuery.

    Args:
        df: Dataframe to be loaded.

    Returns:
        str: Table ID where the data is loaded.
    """
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    return table_id
