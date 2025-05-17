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

from typing import Optional

from steps import model_deployer, model_promoter, model_register, model_trainer
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def training(
    target: Optional[str] = "target",
):
    """Model training pipeline.

    This is a pipeline that loads the data from a preprocessing pipeline,
    trains a model on it and evaluates the model. If it is the first model
    to be trained, it will be promoted to production. If not, it will be
    promoted only if it has a higher accuracy than the current production
    model version.

    Args:
        train_dataset_id: ID of the train dataset produced by feature engineering.
        test_dataset_id: ID of the test dataset produced by feature engineering.
        target: Name of target column in dataset.
        model_type: The type of model to train.
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    _, accuracy = model_trainer(target=target)
    is_promoted = model_promoter(accuracy=accuracy)
    model_registry_uri = model_register(is_promoted)
    model_deployer(
        model_registry_uri=model_registry_uri, is_promoted=is_promoted
    )
