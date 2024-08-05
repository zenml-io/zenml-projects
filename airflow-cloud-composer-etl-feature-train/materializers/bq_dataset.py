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

from typing import Optional

import pandas as pd
from google.cloud import bigquery

from materializers.dataset import Dataset


class BigQueryDataset(Dataset):
    def __init__(
        self,
        table_id: str,
        write_disposition: str = "WRITE_TRUNCATE",
        project: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        self.table_id = table_id
        self.write_disposition = write_disposition
        self.project = project
        self.dataset = dataset
        self.client = bigquery.Client(project=project)

    def fetch_data(self) -> pd.DataFrame:
        query = f"""
        SELECT * FROM `{self.table_id}`
        """
        return self.client.query(query).to_dataframe()

    def load_data(self, df: pd.DataFrame) -> None:
        job_config = bigquery.LoadJobConfig(
            write_disposition=self.write_disposition
        )
        job = self.client.load_table_from_dataframe(
            df, self.table_id, job_config=job_config
        )
        job.result()  # Wait for the job to complete

    def execute_query(self, query: str) -> pd.DataFrame:
        return self.client.query(query).to_dataframe()
