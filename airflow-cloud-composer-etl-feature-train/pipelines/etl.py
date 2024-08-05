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

from materializers.dataset import Dataset
from steps import (
    extract_data_local,
    extract_data_remote,
    transform_bq,
    transform_csv,
)
from zenml import pipeline


@pipeline
def etl_pipeline(mode: str = "develop") -> Dataset:
    """Model deployment pipeline.

    This is a pipeline that loads data to BigQuery.

    Args:
        mode: str: The mode in which the pipeline is run. Defaults to "develop".

    Returns:
        Dataset: The transformed data.
    """
    if mode == "develop":
        raw_data = extract_data_local()
        transformed_data = transform_csv(raw_data)
    else:
        raw_data = extract_data_remote()
        transformed_data = transform_bq(raw_data)
    return transformed_data
