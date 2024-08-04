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

from steps import (
    augment_data,
    load_data_bq,
    load_data_local,
    load_latest_data_bq,
    load_latest_data_local,
)
from zenml import pipeline


@pipeline
def feature_engineering_pipeline(
    data_path: str = "tmp/transformed_data.csv", mode: str = "develop"
):
    """A pipeline to augment data and load it into BigQuery or locally.

    Args:
        data_path: str: The path to the data. Defaults to "tmp/transformed_data.csv".
        mode: str: The mode in which the pipeline is run. Defaults to "develop".

    Returns:
        str: The path to the data.
    """
    if mode == "develop":
        raw_data = load_latest_data_local(data_path)
        augmented_data = augment_data(raw_data)
        data_path = load_data_local(augmented_data, "augmented_data.csv")
    else:
        raw_data = load_latest_data_bq(table_id=data_path)
        augmented_data = augment_data(raw_data)
        data_path = load_data_bq(augmented_data)

    return data_path
