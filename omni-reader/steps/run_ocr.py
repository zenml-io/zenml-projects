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
"""This module contains a unified OCR step that works with multiple models."""

from typing import Dict, List, Optional

import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from utils import (
    MODEL_CONFIGS,
    process_images_with_model,
)

logger = get_logger(__name__)


@step(enable_cache=False)
def run_ocr(
    images: List[str],
    model_name: str,
    custom_prompt: Optional[str] = None,
    running_from_ui: bool = False,
) -> Annotated[pl.DataFrame, "ocr_results"]:
    """Extract text from images using the specified model.

    Args:
        images: List of paths to image files
        model_name: Name of the model to use (e.g., "gpt-4o-mini", "ollama/gemma3:27b", "pixtral-12b-2409")
        custom_prompt: Optional custom prompt to override the default prompt
        running_from_ui: Whether the pipeline is running from the UI (affects logging)

    Returns:
        Dict: Containing results dataframe with OCR results

    Raises:
        ValueError: If the model_name is not supported
    """
    if model_name not in MODEL_CONFIGS:
        supported_models = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unsupported model: {model_name}. Supported models are: {supported_models}"
        )

    model_config = MODEL_CONFIGS[model_name]

    logger.info(f"Running OCR with model: {model_name}")
    logger.info(f"Processing {len(images)} images")

    results_df = process_images_with_model(
        model_config=model_config,
        images=images,
        custom_prompt=custom_prompt,
        running_from_ui=running_from_ui,
    )

    return results_df
