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
    augment_bq,
    augment_csv,
)
from zenml import pipeline


@pipeline
def feature_engineering_pipeline(mode: str = "develop"):
    """A pipeline to augment data and load it into BigQuery or locally.

    Args:
        data_path: str: The path to the data. Defaults to "tmp/transformed_data.csv".
        mode: str: The mode in which the pipeline is run. Defaults to "develop".

    Returns:
        str: The path to the data.
    """
    if mode == "develop":
        augmented_data = augment_csv(transformed_dataset)
    else:
        augmented_data = augment_bq(transformed_dataset)

    return augmented_data
