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

import asyncio
import time
from typing import Any, Dict

import numpy as np
import torch
from datasets import Dataset
from schemas import zenml_project
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import (
    ClaudeEvaluator,
    compute_classification_metrics,
    flatten_metrics,
)
from zenml import log_metadata, step


@step(model=zenml_project)
def compare_models(
    test_dataset: Dataset,
    anthropic_api_key: str,
    modernbert_path: str,
    tokenizer_path: str,
    modernbert_batch_size: int = 10,
    claude_batch_size: int = 2,
    claude_haiku_token_costs: Dict[str, float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare the performance of ModernBERT and Claude Haiku on a test dataset.

    Args:
        test_dataset: Dataset containing test data
        modernbert_path: Path to ModernBERT model
        tokenizer_path: Path to tokenizer
        modernbert_batch_size: Batch size for ModernBERT
        claude_batch_size: Batch size for Claude Haiku

    Returns:
        Dictionary containing performance metrics for ModernBERT and Claude Haiku
    """

    claude_input_cost_per_1k = claude_haiku_token_costs["input_cost_per_1k"]
    claude_output_cost_per_1k = claude_haiku_token_costs["input_cost_per_1k"]

    # Initialize models
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(modernbert_path)
    model.eval()

    # Get texts and labels
    texts = [record["text"] for record in test_dataset]
    labels = test_dataset["label"]
    total_samples = len(texts)

    # ModernBERT evaluation
    with torch.no_grad():
        all_logits = []
        total_modernbert_latency = 0

        for i in range(0, total_samples, modernbert_batch_size):
            batch_texts = texts[i : i + modernbert_batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            start_time = time.time()
            outputs = model(**inputs)
            batch_latency = time.time() - start_time

            all_logits.extend(outputs.logits.numpy())
            total_modernbert_latency += batch_latency

    modernbert_latency = total_modernbert_latency / total_samples

    # Claude evaluation
    evaluator = ClaudeEvaluator(
        batch_size=claude_batch_size,
        api_key=anthropic_api_key,
    )

    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        claude_results = loop.run_until_complete(
            evaluator.evaluate_batch(texts)
        )
    finally:
        loop.close()

    # Filter valid results
    valid_claude_results = [r for r in claude_results if r.error is None]
    valid_indices = [
        i for i, r in enumerate(claude_results) if r.error is None
    ]
    valid_labels = np.array([labels[i] for i in valid_indices])

    # Calculate Claude costs per prediction
    claude_costs = []
    for result in valid_claude_results:
        input_cost = (result.input_tokens / 1000) * claude_input_cost_per_1k
        output_cost = (result.output_tokens / 1000) * claude_output_cost_per_1k
        total_cost = input_cost + output_cost
        claude_costs.append(total_cost)

    # Calculate average cost per prediction and scale to cost per 1000 predictions
    avg_claude_cost = np.mean(claude_costs)
    claude_cost_per_1000 = avg_claude_cost * 1000

    # Prepare ModernBERT metrics
    modernbert_logits = np.array([all_logits[i] for i in valid_indices])

    # Prepare Claude metrics (with dummy logits for binary classification)
    claude_predictions = np.array([r.prediction for r in valid_claude_results])
    claude_logits = np.zeros((len(claude_predictions), 2))
    claude_logits[range(len(claude_predictions)), claude_predictions] = 1

    # Compute performance metrics
    modernbert_performance = compute_classification_metrics(
        (modernbert_logits, valid_labels)
    )
    claude_performance = compute_classification_metrics(
        (claude_logits, valid_labels)
    )

    metrics = {
        "modernbert": {
            "performance": {
                k: float(v) for k, v in modernbert_performance.items()
            },
            "avg_latency": float(modernbert_latency),
            "cost_per_1000": float(
                modernbert_latency * 0.004 * (1000 / modernbert_batch_size)
            ),
        },
        "claude": {
            "performance": {
                k: float(v) for k, v in claude_performance.items()
            },
            "avg_latency": float(
                np.mean([r.latency for r in valid_claude_results])
            ),
            "tokens": {
                k: float(
                    np.mean(
                        [
                            getattr(r, f"{k}_tokens")
                            for r in valid_claude_results
                        ]
                    )
                )
                for k in ["input", "output", "total"]
            },
            "error_rate": float(
                len([r for r in claude_results if r.error is not None])
                / len(claude_results)
            ),
            "cost_per_1000": float(claude_cost_per_1000),
        },
    }

    flat_metrics = {
        "samples_processed": len(valid_indices),
        **flatten_metrics(metrics["modernbert"], "modernbert_"),
        **flatten_metrics(metrics["claude"], "claude_"),
    }

    log_metadata(metadata=flat_metrics, infer_model=True)
    log_metadata(metadata=flat_metrics)

    return metrics
