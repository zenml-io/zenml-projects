# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

import time
from typing import Any, Dict

import numpy as np
import psutil
import torch
from evaluate import load


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    """
    Computes classification metrics from model predictions.

    Args:
        eval_pred: Tuple of (logits, labels) from model evaluation

    Returns:
        Dict of metrics including accuracy, f1, precision, recall and roc_auc
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    return {
        "accuracy": load("accuracy").compute(predictions=predictions, references=labels)[
            "accuracy"
        ],
        "f1": load("f1").compute(predictions=predictions, references=labels, average="weighted")[
            "f1"
        ],
        "precision": load("precision").compute(
            predictions=predictions, references=labels, average="weighted"
        )["precision"],
        "recall": load("recall").compute(
            predictions=predictions, references=labels, average="weighted"
        )["recall"],
        "roc_auc": load("roc_auc").compute(prediction_scores=probs[:, 1], references=labels)[
            "roc_auc"
        ],
    }


def estimate_inference_cost(
    batch_size: int,
    avg_latency: float,
    cost_per_hour: float = 0.5,
) -> float:
    """
    Estimates cost per thousand predictions based on batch size and latency.

    Args:
        batch_size: Number of samples per batch
        avg_latency: Average inference time per batch in seconds
        cost_per_hour: Infrastructure cost per hour

    Returns:
        Cost per thousand predictions in dollars
    """
    predictions_per_hour = (3600 / avg_latency) * batch_size
    cost_per_prediction = cost_per_hour / predictions_per_hour
    return float(cost_per_prediction * 1000)


def calculate_prediction_costs(trainer, avg_latency: float) -> float:
    """
    Calculates total prediction costs for a training run.

    Args:
        trainer: Hugging Face Trainer instance
        avg_latency: Average inference latency per batch

    Returns:
        Total prediction cost in dollars
    """
    inference_cost = estimate_inference_cost(
        batch_size=trainer.args.per_device_eval_batch_size,
        avg_latency=avg_latency,
    )
    return float(inference_cost)


def measure_inference_latency(
    trainer,
    dataset,
    num_runs: int = 3,
) -> Dict[str, float]:
    """
    Measures model inference latency across multiple runs.

    Args:
        trainer: Hugging Face Trainer instance
        dataset: Evaluation dataset
        num_runs: Number of measurement runs

    Returns:
        Dict with mean inference latency and standard deviation
    """
    latencies = []
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = trainer.predict(dataset)
        latencies.append((time.time() - start_time) / len(dataset))

    return {
        "inference_latency": float(np.mean(latencies)),
        "latency_std": float(np.std(latencies)),
    }


def calculate_memory_usage() -> float:
    """
    Calculates current process memory usage.

    Returns:
        Memory usage in megabytes
    """
    memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return float(memory_usage_mb)


def flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flattens nested metrics dictionary into single level.

    Args:
        metrics: Nested metrics dictionary
        prefix: String prefix for flattened keys

    Returns:
        Flattened metrics dictionary
    """
    flattened = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, f"{prefix}{key}_"))
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened
