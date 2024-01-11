#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

"""Model training steps used to train a model on the training data."""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from zenml.client import Client
from zenml.steps import BaseParameters, Output, step
from sklearn.tree import DecisionTreeClassifier

from steps.data_loaders import DATASET_TARGET_COLUMN_NAME
from utils.tracker_helper import enable_autolog, get_tracker_name


class SVCTrainerParams(BaseParameters):
    """Parameters for the SVC trainer step with various hyperparameters.

    Attributes:
        random_state: The random state used for reproducibility. Pass an int for
            reproducible and cached output across multiple step runs.
        C: Penalty parameter C of the error term.
        kernel: Specifies the kernel type to be used in the algorithm.
        degree: Degree of the polynomial kernel function.
        coef0: Independent term in kernel function.
        shrinking: Whether to use the shrinking heuristic.
        probability: Whether to enable probability estimates.
        extra_hyperparams: Extra hyperparameters to pass to the model
            initializer.
    """

    random_state: int = 42
    C: float = 1.320498
    kernel: str = "rbf"
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = False
    extra_hyperparams: dict = {}


@step(
    experiment_tracker=get_tracker_name(),
)
def svc_trainer(
    params: SVCTrainerParams,
    train_dataset: pd.DataFrame,
) -> Output(model=ClassifierMixin, accuracy=float):
    """Train and logs a sklearn C-support vector classification model.
    
    If the experiment tracker is enabled, the model and the training accuracy
    will be logged to the experiment tracker.

    Args:
        params: The hyperparameters for the model.
        train_dataset: The training dataset to train the model on.

    Returns:
        The trained model and the training accuracy.
    """
    enable_autolog()

    X = train_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = train_dataset[DATASET_TARGET_COLUMN_NAME]
    model = SVC(
        C=params.C,
        kernel=params.kernel,
        degree=params.degree,
        coef0=params.coef0,
        shrinking=params.shrinking,
        probability=params.probability,
        random_state=params.random_state,
        **params.extra_hyperparams,
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc}")
    return model, train_acc


class DecisionTreeTrainerParams(BaseParameters):
    """Parameters for the decision tree trainer step with various
    hyperparameters.

    Attributes:
        random_state: The random state used for reproducibility. Pass an int for
            reproducible and cached output across multiple step runs.
        max_depth: The maximum depth of the tree.
        extra_hyperparams: Extra hyperparameters to pass to the model
            initializer.
    """

    random_state: int = 42
    max_depth: int = 5
    extra_hyperparams: dict = {}


@step(
    experiment_tracker=get_tracker_name(),
)
def decision_tree_trainer(
    params: DecisionTreeTrainerParams,
    train_dataset: pd.DataFrame,
) -> Output(model=ClassifierMixin, accuracy=float):
    """Train a sklearn decision tree classifier.

    If the experiment tracker is enabled, the model and the training accuracy
    will be logged to the experiment tracker.

    Args:
        params: The hyperparameters for the model.
        train_dataset: The training dataset to train the model on.
    
    Returns:
        The trained model and the training accuracy.
    """
    enable_autolog()

    X = train_dataset.drop(columns=[DATASET_TARGET_COLUMN_NAME])
    y = train_dataset[DATASET_TARGET_COLUMN_NAME]
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=params.random_state,
        **params.extra_hyperparams,
    )

    model.fit(X, y)
    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc}")
    return model, train_acc
