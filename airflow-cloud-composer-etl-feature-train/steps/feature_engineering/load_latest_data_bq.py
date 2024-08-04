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
def load_latest_data_bq(
    table_id: str = "your-project.your_dataset.ecb_raw_data",
) -> pd.DataFrame:
    """Load the latest data from the data source.

    Args:
        table_id: Table ID to load the data from.

    Returns:
        pd.Datafram: Dataframe containing the data.
    """
    client = bigquery.Client()
    query = f"""
    SELECT * FROM {table_id}
    WHERE load_timestamp = (SELECT MAX(load_timestamp) FROM `your-project.your_dataset.ecb_raw_data`)
    """
    return client.query(query).to_dataframe()
