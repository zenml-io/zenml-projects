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

import os
from typing import Dict, List, Optional

import polars as pl
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from utils.model_configs import MODEL_CONFIGS
from utils.ocr_processing import process_images_with_model

logger = get_logger(__name__)


@step(enable_cache=False)
def run_ocr(
    images: List[str], model_names: List[str], custom_prompt: Optional[str] = None
) -> Annotated[Dict[str, pl.DataFrame], "ocr_results"]:
    """Extract text from images using multiple models in parallel.

    Args:
        images: List of paths to image files
        model_names: List of model names to use
        custom_prompt: Optional custom prompt to override the default prompt

    Returns:
        Dict: Mapping of model name to results dataframe with OCR results

    Raises:
        ValueError: If any model_name is not supported
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    # Validate all models
    for model_name in model_names:
        if model_name not in MODEL_CONFIGS:
            supported_models = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Unsupported model: {model_name}. Supported models are: {supported_models}"
            )

    logger.info(f"Running OCR with {len(model_names)} models: {', '.join(model_names)}")
    logger.info(f"Processing {len(images)} images")

    results = {}

    with ThreadPoolExecutor(max_workers=min(len(model_names), 5)) as executor:
        futures = {
            model_name: executor.submit(
                process_images_with_model,
                model_config=MODEL_CONFIGS[model_name],
                images=images,
                custom_prompt=custom_prompt,
            )
            for model_name in model_names
        }

        with tqdm(total=len(model_names), desc="Processing models") as pbar:
            for model_name, future in futures.items():
                try:
                    results_df = future.result()
                    results[model_name] = results_df
                    logger.info(f"Completed processing with model: {model_name}")
                except Exception as e:
                    logger.error(f"Error processing model {model_name}: {str(e)}")
                    # empty dataframe with error message to avoid pipeline failure
                    results[model_name] = pl.DataFrame(
                        {
                            "id": range(len(images)),
                            "image_name": [os.path.basename(img) for img in images],
                            "raw_text": [f"Error processing with {model_name}: {str(e)}"]
                            * len(images),
                            "processing_time": [0.0] * len(images),
                            "confidence": [0.0] * len(images),
                            "error": [str(e)] * len(images),
                        }
                    )
                finally:
                    pbar.update(1)

    return results
