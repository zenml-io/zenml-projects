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
from zenml.client import Client


@pipeline
def ecb_predictor_model_training_pipeline(
    augmented_dataset_id, mode: str = "develop"
):
    """A pipeline to train an XGBoost model and promote it.

    Args:
        augmented_dataset_id: str: The ID of the augmented dataset.
        mode: str: The mode in which the pipeline is run. Defaults to "develop
    """
    augmented_data = Client().get_artifact_version(augmented_dataset_id)
    _, metrics = train_xgboost_model(augmented_data)
    promote_model(metrics)
