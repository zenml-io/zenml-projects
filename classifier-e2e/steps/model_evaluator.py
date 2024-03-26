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

import pandas as pd
import wandb
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from zenml import (
    get_step_context,
    log_artifact_metadata,
    log_model_metadata,
    step,
)
from zenml.client import Client
from zenml.exceptions import StepContextError
from zenml.logger import get_logger

logger = get_logger(__name__)

et = Client().active_stack.experiment_tracker


@step(experiment_tracker=et.name)
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    target: Optional[str] = "target",
) -> float:
    """Evaluate a trained model.

    This is an example of a model evaluation step that takes in a model artifact
    previously trained by another step in your pipeline, and a training
    and validation data set pair which it uses to evaluate the model's
    performance. The model metrics are then returned as step output artifacts
    (in this case, the model accuracy on the train and test set).

    The suggested step implementation also outputs some warnings if the model
    performance does not meet some minimum criteria. This is just an example of
    how you can use steps to monitor your model performance and alert you if
    something goes wrong. As an alternative, you can raise an exception in the
    step to force the pipeline run to fail early and all subsequent steps to
    be skipped.

    This step is parameterized to configure the step independently of the step code,
    before running it in a pipeline. In this example, the step can be configured
    to use different values for the acceptable model performance thresholds and
    to control whether the pipeline run should fail if the model performance
    does not meet the minimum criteria. See the documentation for more
    information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        model: The pre-trained model artifact.
        dataset_trn: The train dataset.
        dataset_tst: The test dataset.
        min_train_accuracy: Minimal acceptable training accuracy value.
        min_test_accuracy: Minimal acceptable testing accuracy value.
        target: Name of target column in dataset.

    Returns:
        The model accuracy on the test set.
    """
    # Calculate the model accuracy on the train and test set
    trn_acc = model.score(
        dataset_trn.drop(columns=[target]),
        dataset_trn[target],
    )
    tst_acc = model.score(
        dataset_tst.drop(columns=[target]),
        dataset_tst[target],
    )
    logger.info(f"Train accuracy={trn_acc*100:.2f}%")
    logger.info(f"Test accuracy={tst_acc*100:.2f}%")

    messages = []
    if trn_acc < min_train_accuracy:
        messages.append(
            f"Train accuracy {trn_acc*100:.2f}% is below {min_train_accuracy*100:.2f}% !"
        )
    if tst_acc < min_test_accuracy:
        messages.append(
            f"Test accuracy {tst_acc*100:.2f}% is below {min_test_accuracy*100:.2f}% !"
        )
    else:
        for message in messages:
            logger.warning(message)

    predictions = model.predict(dataset_tst.drop(columns=[target]))
    metadata = {
        "train_accuracy": float(trn_acc),
        "test_accuracy": float(tst_acc),
        "confusion_matrix": confusion_matrix(dataset_tst[target], predictions)
        .ravel()
        .tolist(),
    }
    try:
        if get_step_context().model:
            log_model_metadata(metadata={"wandb_url": wandb.run.url})
    except StepContextError:
        # if model not configured not able to log metadata
        pass

    log_artifact_metadata(
        metadata=metadata,
        artifact_name="breast_cancer_classifier",
    )

    wandb.log(
        {
            "confusion_matrix": wandb.sklearn.plot_confusion_matrix(
                dataset_tst[target], predictions, ["No Cancer", "Cancer"]
            )
        }
    )
    wandb.log({"train_accuracy": metadata["train_accuracy"]})
    wandb.log({"test_accuracy": metadata["test_accuracy"]})

    return float(tst_acc)
