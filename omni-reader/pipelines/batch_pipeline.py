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
"""OCR Batch Pipeline implementation for processing images with multiple models."""

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zenml import pipeline
from zenml.logger import get_logger

from steps import (
    load_images,
    run_ocr,
    save_ocr_results,
)

load_dotenv()

logger = get_logger(__name__)


@pipeline
def ocr_batch_pipeline(
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    models: List[str] = None,
    save_ocr_results_data: bool = False,
    ocr_results_output_dir: str = "ocr_results",
) -> None:
    """Run OCR batch processing pipeline with multiple models.

    Args:
        image_paths: Optional list of specific image paths to process
        image_folder: Optional folder to search for images
        custom_prompt: Optional custom prompt to use for the models
        models: List of model names to use for OCR
        save_ocr_results_data: Whether to save OCR results
        ocr_results_output_dir: Directory to save OCR results

    Returns:
        None
    """
    if not models or len(models) == 0:
        raise ValueError("At least one model must be specified for the batch pipeline")

    images = load_images(
        image_paths=image_paths,
        image_folder=image_folder,
    )
    model_results = run_ocr(
        images=images,
        models=models,
        custom_prompt=custom_prompt,
    )

    if save_ocr_results_data:
        save_ocr_results(
            ocr_results=model_results,
            model_names=models,
            output_dir=ocr_results_output_dir,
        )


def run_ocr_batch_pipeline(config: Dict[str, Any]) -> None:
    """Run the OCR batch pipeline from a configuration dictionary.

    Args:
        config: Dictionary containing configuration

    Returns:
        None
    """
    # Check pipeline mode
    mode = config.get("parameters", {}).get("mode", "batch")
    if mode != "batch":
        logger.warning(f"Expected mode 'batch', but got '{mode}'. Proceeding anyway.")

    # Get selected models from config
    selected_models = config.get("parameters", {}).get("selected_models", [])
    if not selected_models:
        raise ValueError(
            "No models selected in configuration. Add 'selected_models' to parameters section."
        )

    # Create pipeline instance
    pipeline_instance = ocr_batch_pipeline.with_options(
        enable_cache=config.get("enable_cache", False),
    )

    # Get params from config
    pipeline_params = config.get("parameters", {})
    pipeline_steps = config.get("steps", {})
    save_ocr_results_params = pipeline_steps.get("save_ocr_results", {}).get("parameters", {})

    # Run the pipeline
    pipeline_instance(
        image_paths=pipeline_params.get("input_image_paths", []),
        image_folder=pipeline_params.get("input_image_folder"),
        custom_prompt=pipeline_steps.get("run_ocr", {}).get("parameters", {}).get("custom_prompt"),
        models=selected_models,
        save_ocr_results_data=save_ocr_results_params.get("save_locally", False),
        ocr_results_output_dir=save_ocr_results_params.get("output_dir", "ocr_results"),
    )
