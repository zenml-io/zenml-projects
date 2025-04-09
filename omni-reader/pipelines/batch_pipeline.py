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

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.logger import get_logger

from steps import (
    load_images,
    run_ocr,
)

load_dotenv()

logger = get_logger(__name__)

docker_settings = DockerSettings(
    python_package_installer="uv",
    requirements="requirements.txt",
    environment={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    },
)


@pipeline(settings={"docker": docker_settings})
def batch_ocr_pipeline(
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    models: List[str] = None,
) -> None:
    """Run OCR batch processing pipeline with multiple models.

    Args:
        image_paths: Optional list of specific image paths to process
        image_folder: Optional folder to search for images
        custom_prompt: Optional custom prompt to use for the models
        models: List of model names to use for OCR
    """
    images = load_images(
        image_paths=image_paths,
        image_folder=image_folder,
    )

    run_ocr(
        images=images,
        models=models,
        custom_prompt=custom_prompt,
    )


def run_batch_ocr_pipeline(config: Dict[str, Any]) -> None:
    """Run the OCR batch pipeline from a configuration dictionary.

    Args:
        config: Dictionary containing configuration

    Returns:
        None
    """
    pipeline_instance = batch_ocr_pipeline.with_options(
        enable_cache=config.get("enable_cache", False),
    )

    load_images_params = config.get("steps", {}).get("load_images", {}).get("parameters", {})
    image_folder = load_images_params.get("image_folder")
    image_paths = load_images_params.get("image_paths", [])
    if not image_folder and len(image_paths) == 0:
        raise ValueError("Either image_folder or image_paths must be provided")

    run_ocr_params = config.get("steps", {}).get("run_ocr", {}).get("parameters", {})
    custom_prompt = run_ocr_params.get("custom_prompt")
    selected_models = run_ocr_params.get("models", [])
    if not selected_models or len(selected_models) == 0:
        raise ValueError(
            "No models found in the run_ocr step of the batch_ocr_pipeline config file. At least one model must be specified in the 'models' parameter."
        )

    pipeline_instance(
        image_paths=image_paths,
        image_folder=image_folder,
        custom_prompt=custom_prompt,
        models=selected_models,
    )
