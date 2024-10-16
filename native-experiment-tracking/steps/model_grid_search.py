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
import random
from typing import Optional, Tuple

import pandas as pd
from io import BytesIO

from sklearn.base import ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing_extensions import Annotated
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)

@step(enable_cache=False)
def model_grid_searcher(
        dataset_trn: pd.DataFrame,
        target: Optional[str] = "target",
) -> Tuple[
    Annotated[
        ClassifierMixin,
        ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
    ],
    Annotated[
        Image.Image, "grid_search_results"
    ],
    Annotated[
        Image.Image, "confusion_matrix"
    ]
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    alpha_values = [0.0001, 0.001, 0.01, 0.1]
    penalties = ["l2", "l1", "elasticnet"]
    losses = ["hinge", "log", "squared_hinge", "modified_huber", "perceptron"]
    params = {
        "alpha": alpha_values,
        "penalty": penalties,
        "loss": losses,
    }

    clf = SGDClassifier(max_iter=random.randint(10, 1500))
    grid = GridSearchCV(clf, param_grid=params, cv=10)

    grid.fit(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target]
    )
    logger.info(f"Training model {clf}...")

    img = generate_plot(alpha_values, grid, losses, penalties)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(dataset_trn.drop(columns=[target]))

    cm = confusion_matrix(dataset_trn[target], y_pred)

    cm_img = generate_cm(best_model, cm)

    return grid.best_estimator_, img, cm_img


def generate_cm(best_model, cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    # Convert the BytesIO object to a PIL Image
    buf.seek(0)
    cm_img = Image.open(buf)
    return cm_img


def generate_plot(alpha_values, grid, losses, penalties):
    best_params = grid.best_params_
    logger.info(f"Best params: {best_params}")
    best_alpha_idx = alpha_values.index(best_params['alpha'])
    best_penalty_idx = penalties.index(best_params['penalty'])
    best_loss_idx = losses.index(best_params['loss'])
    scores_mean = grid.cv_results_['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(alpha_values), len(losses), len(penalties))
    fig, axes = plt.subplots(len(penalties), 1, figsize=(12, 4 * len(penalties)), sharex=True)
    fig.suptitle("Grid Search Scores", fontsize=20, fontweight='bold')

    for idx, val in enumerate(penalties):
        ax = axes[idx] if len(penalties) > 1 else axes
        for i, p2_val in enumerate(losses):
            ax.plot(alpha_values, scores_mean[:, i, idx], '-o', label=f"loss: {p2_val}")

            # Highlight the best param combination
            if idx == best_penalty_idx and i == best_loss_idx:
                ax.plot(alpha_values[best_alpha_idx], scores_mean[best_alpha_idx, i, idx],
                        'ro', markersize=12, markerfacecolor='none', markeredgewidth=2,
                        markeredgecolor='r', label='Best Estimator')

        ax.set_title(f"penalty: {val}", fontsize=16)
        ax.set_xlabel("alpha", fontsize=14)
        ax.set_xscale('log')
        ax.set_ylabel('CV Average Score', fontsize=14)
        ax.legend(loc="best", fontsize=12)
        ax.grid(True)
    # plt.tight_layout()
    # plt.show()
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    # Convert the BytesIO object to a PIL Image
    buf.seek(0)
    img = Image.open(buf)
    return img

