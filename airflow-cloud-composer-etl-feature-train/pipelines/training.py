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
    promote_model,
    train_xgboost_model,
)
from zenml import pipeline


@pipeline
def model_training_pipeline(
    data_path: str = "tmp/augmented_data.csv", mode: str = "develop"
):
    """A pipeline to train an XGBoost model and promote it.

    Args:
        data_path: str: The path to the data. Defaults to "tmp/augmented_data.csv".
        mode: str: The mode in which the pipeline is run. Defaults to "develop
    """
    _, metrics = train_xgboost_model(augmented_data)
    promote_model(metrics)
