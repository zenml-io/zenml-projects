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

from steps import model_evaluator, model_trainer
from zenml import pipeline
from zenml.logger import get_logger

from pipelines import (
    feature_engineering,
)

logger = get_logger(__name__)


@pipeline
def training(
    alpha_value: float,
    penalty: str,
    loss: str,
    target: Optional[str] = "target",
):
    """
    Model training pipeline.

    This is a pipeline that loads the data from a preprocessing pipeline,
    trains a model on it and evaluates the model. If it is the first model
    to be trained, it will be promoted to production. If not, it will be
    promoted only if it has a higher accuracy than the current production
    model version.

    Args:
        target: Name of target column in dataset.
        alpha_value: Alpha value to use for the train step,
        penalty: Penalty to use for sgd,
        loss: Loss function to be used for sgd,
    """
    # Link all the steps together by calling them and passing the output
    # of one step as the input of the next step.

    # Execute Feature Engineering Pipeline
    dataset_trn, dataset_tst = feature_engineering()

    model, _ = model_trainer(
        dataset_trn=dataset_trn,
        target=target,
        alpha_value=alpha_value,
        penalty=penalty,
        loss=loss,
    )

    test_acc = model_evaluator(
        model=model,
        dataset_trn=dataset_trn,
        dataset_tst=dataset_tst,
        target=target,
    )
    return test_acc
