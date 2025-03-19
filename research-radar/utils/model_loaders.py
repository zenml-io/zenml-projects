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
Utilities for loading models with appropriate settings.
"""

from typing import Dict, Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from utils import logger


def load_base_model(
    base_model_id: str,
    num_labels: int = 2,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
    device: str = "auto",
    remote_execution: bool = False,
) -> AutoModelForSequenceClassification:
    """Load ModernBERT model with appropriate device and precision settings.

    Args:
        base_model_id: HuggingFace model ID
        num_labels: Number of classification labels
        label2id: Mapping from label names to IDs
        id2label: Mapping from IDs to label names
        device: Device to load model on ('auto', 'cuda', 'mps', 'cpu')
        remote_execution: Whether to use optimized settings for remote execution

    Returns:
        Loaded model
    """
    if label2id is None:
        label2id = {"negative": 0, "positive": 1}
    if id2label is None:
        id2label = {0: "negative", 1: "positive"}

    model_kwargs = {
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": id2label,
    }

    # Apply device-specific optimizations
    if device == "cuda":
        if remote_execution:
            logger.info(
                "Using optimized GPU configuration for remote execution"
            )
            if (
                torch.cuda.is_available()
                and torch.cuda.get_device_properties(0).total_memory > 10e9
            ):  # > 10GB memory
                model_kwargs["torch_dtype"] = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
                logger.info(
                    f"Using mixed precision: {model_kwargs.get('torch_dtype')}"
                )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id, **model_kwargs
    )

    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")

    return model


def load_tokenizer(
    base_model_id: str, use_fast: bool = True
) -> PreTrainedTokenizerFast:
    """Load tokenizer for the model.

    Args:
        base_model_id: HuggingFace model ID
        use_fast: Whether to use the fast tokenizer implementation

    Returns:
        Loaded tokenizer
    """
    logger.info(f"Loading tokenizer from {base_model_id}")
    return AutoTokenizer.from_pretrained(base_model_id, use_fast=use_fast)
