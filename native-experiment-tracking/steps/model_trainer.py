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
from typing import Optional, List, Tuple

import pandas as pd
import sys
from io import StringIO, BytesIO
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from typing_extensions import Annotated
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from zenml import ArtifactConfig, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_trainer(
    dataset_trn: pd.DataFrame,
    model_type: str = "sgd",
    target: Optional[str] = "target",
) -> Tuple[
        Annotated[
            ClassifierMixin,
            ArtifactConfig(name="sklearn_classifier", is_model_artifact=True)
        ],
        Annotated[
            List[float], "training_loss"
        ],
        Annotated[
            Image.Image, "training_loss_graph"
        ]
]:
    """Configure and train a model on the training dataset.

    This is an example of a model training step that takes in a dataset artifact
    previously loaded and pre-processed by other steps in your pipeline, then
    configures and trains a model on it. The model is then returned as a step
    output artifact.

    Args:
        dataset_trn: The preprocessed train dataset.
        model_type: The type of model to train.
        target: The name of the target column in the dataset.

    Returns:
        The trained model artifact.

    Raises:
        ValueError: If the model type is not supported.
    """
    # Initialize the model with the hyperparameters indicated in the step
    # parameters and train it on the training set.
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()


    model = SGDClassifier(verbose=1)
    logger.info(f"Training model {model}...")

    model.fit(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )

    sys.stdout = old_stdout

    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if (len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))

    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Time in epochs")
    plt.ylabel("Loss")

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()

    # Convert the BytesIO object to a PIL Image
    buf.seek(0)
    img = Image.open(buf)

    return model, loss_list, img
