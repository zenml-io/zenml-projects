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
from io import BytesIO
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.base import ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils._param_validation import InvalidParameterError
from typing_extensions import Annotated
from zenml import (
    ArtifactConfig,
    get_step_context,
    log_artifact_metadata,
    log_model_metadata,
    step,
)
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def model_trainer(
    dataset_trn: pd.DataFrame,
    alpha_value: float,
    penalty: str,
    loss: str,
    target: Optional[str] = "target",
) -> Tuple[
    Annotated[
        ClassifierMixin,
        ArtifactConfig(name="sklearn_classifier", is_model_artifact=True),
    ],
    Annotated[Image.Image, "confusion_matrix"],
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        target: The name of the target column in the dataset.
        alpha_value: Alpha value to use for the train step,
        penalty: Penalty to use for sgd,
        loss: Loss function to be used for sgd,

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    client = Client()

    log_model_metadata(
        metadata={
            "alpha_value": alpha_value,
            "penalty": penalty,
            "loss": loss,
        }
    )
    log_artifact_metadata(
        metadata={
            "alpha_value": alpha_value,
            "penalty": penalty,
            "loss": loss,
        },
        artifact_name="sklearn_classifier",
    )

    model = SGDClassifier(
        max_iter=1000, penalty=penalty, alpha=alpha_value, loss=loss
    )

    try:
        model.fit(dataset_trn.drop(columns=[target]), dataset_trn[target])
    except InvalidParameterError:
        logger.info(
            "Invalid parameter combination: alpha: {alpha_value}, penalty: {penalty}, loss: {loss}!\n\n"
        )

        model = get_step_context().model
        client.delete_model_version(model_version_id=model.model_version_id)
        raise InvalidParameterError(
            f"Invalid parameter combination: alpha: {alpha_value}, penalty: {penalty}, loss: {loss}!\n\n"
        )

    logger.info(f"Training model {model}...")

    y_pred = model.predict(dataset_trn.drop(columns=[target]))

    cm = confusion_matrix(dataset_trn[target], y_pred)

    cm_img = generate_cm(model, cm)

    return model, cm_img


def generate_cm(best_model, cm):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=best_model.classes_
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    # Convert the BytesIO object to a PIL Image
    buf.seek(0)
    cm_img = Image.open(buf)
    return cm_img
