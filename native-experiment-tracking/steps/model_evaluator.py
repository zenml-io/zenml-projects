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
from typing import Optional, Tuple, Annotated

import pandas as pd
from PIL import Image
from sklearn.base import ClassifierMixin
import matplotlib.pyplot as plt

from zenml import step, log_model_metadata, get_step_context
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def model_evaluator(
    model: ClassifierMixin,
    dataset_trn: pd.DataFrame,
    dataset_tst: pd.DataFrame,
    min_train_accuracy: float = 0.0,
    min_test_accuracy: float = 0.0,
    target: Optional[str] = "target",
) -> Tuple[
    Annotated[
        float,
        "test_accuracy"
    ],
    Annotated[
        Image.Image,
        "accuracy_history"
    ]
]:
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

        https://docs.zenml.io/how-to/build-pipelines/use-pipeline-step-parameters

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

    client = Client()
    mv = get_step_context().model
    print("got mv")
    m = client.list_model_versions(model_name_or_id=mv.model_id, size=1)
    print("got m")
    number_of_versions = m.total
    print(f"num mv {number_of_versions}")

    # Initialize a list to store test accuracies
    versions = [str(mv.number)]
    test_accuracies = [tst_acc]

    # Start with the latest version and iterate backwards
    index = number_of_versions - 1

    # Fetch test accuracies until we have 10 versions or reach the start
    while len(versions) < 15 and index > 0:
        print(f"getting index: {index}")
        zenml_model_version = client.get_model_version("breast_cancer_classifier", index, hydrate=False)
        if zenml_model_version.run_metadata:
            test_accuracy = zenml_model_version.run_metadata['test_accuracy'].value
            versions.append(str(zenml_model_version.number))
            test_accuracies.append(test_accuracy)

        index -= 1  # Move to the previous version

    # Find the index of the model version in the version list
    if mv.version in versions:
        highlight_index = versions.index(mv.version)
    else:
        highlight_index = None
        print(f"{mv.version} not found in version list.")

    chronological_versions = versions[::-1]
    chronological_test_accuracies = test_accuracies[::-1]

    # Create a line plot for the test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(chronological_versions, chronological_test_accuracies, marker='o', linestyle='-', color='b', label='Test Accuracy')

    # If the highlight_version was found, highlight the specific point
    if highlight_index is not None:
        plt.scatter(versions[highlight_index], chronological_test_accuracies[highlight_index], color='red', s=100, zorder=5,
                    label='Current Run')
        plt.annotate(f'{versions[highlight_index]}: {chronological_test_accuracies[highlight_index]:.2f}',
                     (versions[highlight_index], chronological_test_accuracies[highlight_index]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')

    # Customize axis labels and titles
    plt.xlabel('Model Version')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy for the Last 10 Runs')
    plt.xticks(chronological_versions)  # Display model versions on the x-axis
    plt.ylim(0, 1)  # Assuming test accuracy is between 0 and 1
    plt.grid(True)  # Add a grid for better readability
    plt.legend(loc='lower right')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    # Convert the BytesIO object to a PIL Image
    buf.seek(0)
    history = Image.open(buf)

    log_model_metadata(
        metadata={
            "train_accuracy": float(trn_acc),
            "test_accuracy": float(tst_acc),
        }
    )
    return float(tst_acc), history
