# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
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

from typing import Dict, List

import numpy as np
from datasets import load_metric


def compute_metrics(eval_pred: tuple) -> Dict[str, float]:
    """Compute the metrics for the model.

    Args:
        eval_pred: The evaluation prediction.

    Returns:
        The metrics for the model.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculate the mertic using the predicted and true value
    accuracy = load_metric("accuracy").compute(
        predictions=predictions, references=labels
    )
    f1 = load_metric("f1").compute(
        predictions=predictions, references=labels, average="weighted"
    )
    precision = load_metric("precision").compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
    }


def find_max_length(dataset: List[str]) -> int:
    """Find the maximum length of the dataset.

    Args:
        dataset: The dataset.

    Returns:
        The maximum length of the dataset.
    """
    return len(max(dataset, key=lambda x: len(x.split())).split())
