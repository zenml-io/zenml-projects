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
from typing import Any, Dict

import numpy as np
from datasets import Dataset
from schemas import zenml_project
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import (
    ClaudeEvaluator,
    calculate_claude_costs,
    evaluate_modernbert,
    flatten_metrics,
    prepare_metrics,
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
        anthropic_api_key: Anthropic API key for Claude
        modernbert_path: Path to ModernBERT model
        tokenizer_path: Path to tokenizer
        modernbert_batch_size: Batch size for ModernBERT
        claude_batch_size: Batch size for Claude Haiku
        claude_haiku_token_costs: Token costs for Claude Haiku

    Returns:
        Dictionary containing performance metrics for ModernBERT and Claude Haiku
    """

    claude_input_cost_per_1k = claude_haiku_token_costs["input_cost_per_1k"]
    claude_output_cost_per_1k = claude_haiku_token_costs["input_cost_per_1k"]

    # initialize models
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(modernbert_path)
    model.eval()

    texts = [record["text"] for record in test_dataset]
    labels = test_dataset["label"]

    # evaluate modernbert
    all_logits, modernbert_latency = evaluate_modernbert(
        model, tokenizer, texts, modernbert_batch_size
    )

    # evaluate claude
    evaluator = ClaudeEvaluator(
        batch_size=claude_batch_size,
        api_key=anthropic_api_key,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        claude_results = loop.run_until_complete(
            evaluator.evaluate_batch(texts)
        )
    finally:
        loop.close()

    # filter valid results
    valid_indices = [
        i for i, r in enumerate(claude_results) if r.error is None
    ]
    valid_labels = np.array([labels[i] for i in valid_indices])
    valid_claude_results = [r for r in claude_results if r.error is None]

    # calculate claude costs
    _, claude_cost_per_1000 = calculate_claude_costs(
        valid_claude_results,
        claude_input_cost_per_1k,
        claude_output_cost_per_1k,
    )

    metrics = prepare_metrics(
        all_logits,
        claude_results,
        valid_indices,
        valid_labels,
        modernbert_latency,
        modernbert_batch_size,
        claude_cost_per_1000,
    )

    flat_metrics = {
        "samples_processed": len(valid_indices),
        **flatten_metrics(metrics["modernbert"], "modernbert_"),
        **flatten_metrics(metrics["claude"], "claude_"),
    }

    log_metadata(metadata=flat_metrics, infer_model=True)
    log_metadata(metadata=flat_metrics)

    return metrics
