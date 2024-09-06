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

import os
from datetime import datetime, timezone
from typing import Optional
from typing_extensions import Annotated

import pandas as pd
from materializers import BigQueryDataset, CSVDataset
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def transform_csv(
    df: pd.DataFrame, filename: str = "transformed_data.csv"
) -> Annotated[CSVDataset, "ecb_transformed_dataset"]:
    """Transform the data by adding a processed column and a load timestamp.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    df["processed"] = 1
    df["load_timestamp"] = datetime.now(timezone.utc).isoformat()

    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    data_path = f"tmp/{filename}"

    return CSVDataset(data_path, df)


@step
def transform_bq(
    df: pd.DataFrame, table_id: str, bq_config: Optional[dict] = {}
) -> Annotated[BigQueryDataset, "ecb_transformed_dataset"]:
    """Transform the data by adding a processed column and a load timestamp.

    Args:
        df: Input dataframe.

    Returns:
        pd.DataFrame: Transformed dataframe.
    """
    df["processed"] = 1
    df["load_timestamp"] = datetime.now(timezone.utc).isoformat()
    return BigQueryDataset(df=df, table_id=table_id, **bq_config)
