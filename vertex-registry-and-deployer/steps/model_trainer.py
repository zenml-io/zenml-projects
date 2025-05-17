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

from typing import Tuple, Union

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Annotated
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger
from zenml.utils.metadata_utils import log_metadata

logger = get_logger(__name__)


@step(enable_cache=False)
def model_trainer(
    random_state: int = 42,
    test_size: float = 0.2,
    drop_na: bool = True,
    normalize: bool = True,
    target: str = "target",
    min_train_accuracy: float = 0.3,
    min_test_accuracy: float = 0.3,
) -> Tuple[
    Annotated[
        ClassifierMixin,
        ArtifactConfig(
            name="sklearn_classifier", artifact_type=ArtifactType.MODEL
        ),
    ],
    Annotated[float, ArtifactConfig(name="accuracy")],
]:
    # Load the dataset
    dataset = load_breast_cancer(as_frame=True).frame
    dataset.reset_index(drop=True, inplace=True)
    logger.info(f"Dataset with {len(dataset)} records loaded!")

    # Split the dataset
    dataset_trn, dataset_tst = train_test_split(
        dataset,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Separate features and target
    X_trn = dataset_trn.drop(columns=[target])
    y_trn = dataset_trn[target]
    X_tst = dataset_tst.drop(columns=[target])
    y_tst = dataset_tst[target]

    # Preprocess the data
    preprocess_steps = []
    if drop_na:
        preprocess_steps.append(("drop_na", NADropper()))
    if normalize:
        preprocess_steps.append(("normalize", MinMaxScaler()))
    preprocess_pipeline = Pipeline(preprocess_steps)

    X_trn = preprocess_pipeline.fit_transform(X_trn)
    X_tst = preprocess_pipeline.transform(X_tst)

    # Train the model
    model = SGDClassifier()
    logger.info(f"Training model {model}...")

    model.fit(X_trn, y_trn)

    # Evaluate the model
    trn_acc = model.score(X_trn, y_trn)
    tst_acc = model.score(X_tst, y_tst)
    logger.info(f"Train accuracy={trn_acc * 100:.2f}%")
    logger.info(f"Test accuracy={tst_acc * 100:.2f}%")

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc * 100:.2f}% is below {min_train_accuracy * 100:.2f}%!"
        )
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc * 100:.2f}% is below {min_test_accuracy * 100:.2f}%!"
        )
    else:
        for message in messages:
            logger.warning(message)

    log_metadata(
        metadata={
            "train_accuracy": float(trn_acc),
            "test_accuracy": float(tst_acc),
        },
        artifact_name="sklearn_classifier",
        infer_artifact=True,
    )
    return model, tst_acc


class NADropper:
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, *args, **kwargs):  # noqa: D102
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):  # noqa: D102
        return X.dropna()
