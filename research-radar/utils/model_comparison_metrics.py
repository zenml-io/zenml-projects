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

"""
Utilities for model comparison operations.
"""

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def evaluate_modernbert(
    model, 
    tokenizer, 
    texts: List[str], 
    batch_size: int
) -> Tuple[List[np.ndarray], float]:
    """
    Evaluate ModernBERT model on input texts.
    
    Args:
        model: The ModernBERT model
        tokenizer: The tokenizer for ModernBERT
        texts: List of text inputs
        batch_size: Batch size for processing
        
    Returns:
        Tuple containing logits and average latency
    """
    total_samples = len(texts)
    all_logits = []
    total_latency = 0
    
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_texts = texts[i : i + batch_size]
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
            total_latency += batch_latency

    avg_latency = total_latency / total_samples
    return all_logits, avg_latency


def calculate_claude_costs(
    results: List[Any], 
    input_cost_per_1k: float, 
    output_cost_per_1k: float
) -> Tuple[List[float], float]:
    """
    Calculate costs for Claude API usage.
    
    Args:
        results: List of Claude results
        input_cost_per_1k: Cost per 1k input tokens
        output_cost_per_1k: Cost per 1k output tokens
        
    Returns:
        Tuple containing list of costs and cost per 1000 predictions
    """
    costs = []
    for result in results:
        input_cost = (result.input_tokens / 1000) * input_cost_per_1k
        output_cost = (result.output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        costs.append(total_cost)

    avg_cost = np.mean(costs)
    cost_per_1000 = avg_cost * 1000
    return costs, cost_per_1000


def prepare_metrics(
    modernbert_logits: List[np.ndarray],
    claude_results: List[Any],
    valid_indices: List[int],
    valid_labels: np.ndarray,
    modernbert_latency: float,
    modernbert_batch_size: int,
    claude_cost_per_1000: float
) -> Dict[str, Dict[str, Any]]:
    """
    Prepare performance metrics for both models.
    
    Args:
        modernbert_logits: ModernBERT logits
        claude_results: Claude API results
        valid_indices: Indices of valid results
        valid_labels: Ground truth labels
        modernbert_latency: Average latency for ModernBERT
        modernbert_batch_size: Batch size for ModernBERT
        claude_cost_per_1000: Cost per 1000 predictions for Claude
        
    Returns:
        Dictionary of metrics for both models
    """
    from utils import compute_classification_metrics
    
    # ModernBERT metrics
    mb_logits = np.array([modernbert_logits[i] for i in valid_indices])
    modernbert_performance = compute_classification_metrics(
        (mb_logits, valid_labels)
    )
    
    # Claude metrics
    valid_claude_results = [r for r in claude_results if r.error is None]
    claude_predictions = np.array([r.prediction for r in valid_claude_results])
    claude_logits = np.zeros((len(claude_predictions), 2))
    claude_logits[range(len(claude_predictions)), claude_predictions] = 1
    claude_performance = compute_classification_metrics(
        (claude_logits, valid_labels)
    )
    
    return {
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