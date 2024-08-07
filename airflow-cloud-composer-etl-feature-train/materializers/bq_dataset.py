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
        df: Optional[pd.DataFrame] = None,
        write_disposition: str = "WRITE_TRUNCATE",
        project: Optional[str] = None,
        dataset: Optional[str] = None,
    ):
        self.table_id = table_id
        self.write_disposition = write_disposition
        self.project = project
        self.dataset = dataset
        if project is None:
            # Split up the table_id to get the project
            self.project = table_id.split(".")[0]
            self.client = bigquery.Client()
        self.client = bigquery.Client(project=self.project)
        self.df = df

    def read_data(self) -> pd.DataFrame:
        query = f"""
        SELECT * FROM `{self.table_id}`
        """
        self.df = self.client.query(query).to_dataframe()
        return self.df

    def write_data(self) -> None:
        job_config = bigquery.LoadJobConfig(
            write_disposition=self.write_disposition
        )
        job = self.client.load_table_from_dataframe(
            self.df, self.table_id, job_config=job_config
        )
        job.result()  # Wait for the job to complete
