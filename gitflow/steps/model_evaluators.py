#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

"""Model scoring and evaluation steps used to check the model performance"""

from typing import List
import pandas as pd
from sklearn.base import ClassifierMixin

from zenml.integrations.deepchecks.steps import (
    DeepchecksModelValidationCheckStepParameters,
    deepchecks_model_validation_check_step,
)
from zenml.integrations.deepchecks.steps import (
    DeepchecksModelDriftCheckStepParameters,
    deepchecks_model_drift_check_step,
)
from zenml.steps import BaseParameters, step

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME
from utils.tracker_helper import get_tracker_name, log_metric


class ModelScorerStepParams(BaseParameters):
    """Parameters for the model scorer step.

    Attributes:
        accuracy_metric_name: The name of the metric used to log the accuracy
            in the experiment tracker.
    """

    accuracy_metric_name: str = "accuracy"


def score_model(
    dataset: pd.DataFrame,
    model: ClassifierMixin,
) -> float:
    """Calculate the model accuracy on a given dataset.

    Args:
        dataset: The dataset to score the model on.
        model: The model to score.

    Returns:
        The accuracy of the model on the dataset.
    """
    X = dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = dataset[DATASET_TARGET_COLUMN_NAME]
    acc = model.score(X, y)
    return acc


@step(
    experiment_tracker=get_tracker_name(),
)
def model_scorer(
    params: ModelScorerStepParams,
    dataset: pd.DataFrame,
    model: ClassifierMixin,
) -> float:
    """Calculate and log the model accuracy on a given dataset.

    If the experiment tracker is enabled, the scoring accuracy
    will be logged to the experiment tracker.

    Args:
        params: The parameters for the model scorer step.
        dataset: The dataset to score the model on.
        model: The model to score.

    Returns:
        The accuracy of the model on the dataset.
    """
    acc = score_model(dataset, model)
    log_metric(params.accuracy_metric_name, acc)
    print(f"{params.accuracy_metric_name}: {acc}")
    return acc


@step(
    experiment_tracker=get_tracker_name(),
)
def optional_model_scorer(
    params: ModelScorerStepParams,
    dataset: pd.DataFrame,
    model: List[ClassifierMixin],
) -> float:
    """Calculate and log the model accuracy on a given dataset with an optional
    model.

    This is a variation of the model_scorer step that can handle an optional
    model. This is useful for cases where the model loaded by a previous step
    is not available, e.g. when the model is not found in the model registry.

    If the experiment tracker is enabled, the scoring accuracy will be logged to
    the experiment tracker.

    Args:
        params: The parameters for the model scorer step.
        dataset: The dataset to score the model on.
        model: Optinoal model to score.

    Returns:
        The accuracy of the model on the dataset. If no model is provided, 0.0
        is returned instead.
    """
    if not len(model):
        return 0.0
    acc = score_model(dataset, model[0])
    log_metric(params.accuracy_metric_name, acc)
    print(f"{params.accuracy_metric_name}: {acc}")
    return acc


# Deepchecks train-test model evaluation step.
train_test_model_evaluator = deepchecks_model_drift_check_step(
    step_name="train_test_model_evaluator",
    params=DeepchecksModelDriftCheckStepParameters(
        dataset_kwargs=dict(label=DATASET_TARGET_COLUMN_NAME, cat_features=[]),
    ),
)

# Deepchecks model evaluation step.
model_evaluator = deepchecks_model_validation_check_step(
    step_name="model_evaluator",
    params=DeepchecksModelValidationCheckStepParameters(
        dataset_kwargs=dict(label=DATASET_TARGET_COLUMN_NAME, cat_features=[]),
    ),
)
