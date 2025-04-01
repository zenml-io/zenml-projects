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
"""OCR Comparison Pipeline implementation with YAML configuration support."""

import os
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.logger import get_logger

from steps import (
    evaluate_models,
    load_ground_truth_file,
    load_images,
    run_ocr,
    save_ocr_results,
    save_visualization,
)

load_dotenv()

docker_settings = DockerSettings(
    dockerfile="Dockerfile",
    requirements=[
        "polars==1.26.0",
        "textdistance==4.6.3",
        "instructor==1.7.7",
        "jiwer==3.0.5",
        "litellm==1.64.1",
        "openai==1.69.0",
        "mistralai==1.5.0",
        "Pillow==11.1.0",
        "ollama==0.4.7",
        "pyarrow>=7.0",
    ],
    environment={
        "OLLAMA_HOST": "${OLLAMA_HOST:-http://localhost:11434}",
        "OLLAMA_MODELS": "/root/.ollama",
        "OLLAMA_TIMEOUT": "600s",
        "MISTRAL_API_KEY": "${MISTRAL_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
    },
)

logger = get_logger(__name__)


@step
def extract_ground_truth_df(ground_truth_results, ground_truth_model):
    """Extract ground truth DataFrame from the results dictionary.

    Args:
        ground_truth_results: Dictionary with model results returned by run_ocr
        ground_truth_model: Name of the ground truth model

    Returns:
        The ground truth DataFrame
    """
    if ground_truth_model in ground_truth_results:
        return ground_truth_results[ground_truth_model]
    return None


@pipeline(settings={"docker": docker_settings})
def ocr_comparison_pipeline(
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    models: Optional[List[str]] = None,
    ground_truth_model: str = "gpt-4o-mini",
    ground_truth_source: Literal["openai", "manual", "file", "none"] = "none",
    ground_truth_file: Optional[str] = None,
    save_ground_truth_data: bool = False,
    ground_truth_output_dir: str = "ocr_results",
    save_ocr_results_data: bool = False,
    ocr_results_output_dir: str = "ocr_results",
    save_visualization_data: bool = False,
    visualization_output_dir: str = "visualizations",
) -> None:
    """Run OCR comparison pipeline between multiple configurable models.

    Args:
        image_paths: Optional list of specific image paths to process
        image_folder: Optional folder to search for images
        custom_prompt: Optional custom prompt to use for both models
        models: List of model names to use (if None, uses default models)
        ground_truth_model: Name of the model to use for ground truth when source is "openai"
        ground_truth_source: Source of ground truth - "openai" to use configured model, "manual" for user-provided texts,
                            "file" to load from a saved JSON file, or "none" to skip ground truth evaluation
        ground_truth_file: Path to ground truth JSON file (used when ground_truth_source="file")
        save_ground_truth_data: Whether to save generated ground truth data for future use
        ground_truth_output_dir: Directory to save ground truth data
        save_ocr_results_data: Whether to save OCR results from both models
        ocr_results_output_dir: Directory to save OCR results
        save_visualization_data: Whether to save HTML visualization to local file
        visualization_output_dir: Directory to save HTML visualization

    Returns:
        None
    """
    images = load_images(image_paths=image_paths, image_folder=image_folder)

    # Default to two models if none provided
    if not models or len(models) < 2:
        models = ["llama3.2-vision:11b", "pixtral-12b-2409"]

    # Process all models in parallel
    model_results = run_ocr(images=images, model_names=models, custom_prompt=custom_prompt)

    # Process ground truth separately to avoid including it in the main comparison
    ground_truth_df = None
    if ground_truth_source == "openai":
        # Run OCR on the ground truth model
        ground_truth_results = run_ocr(
            images=images, model_names=[ground_truth_model], custom_prompt=custom_prompt
        )

        # Extract the ground truth DataFrame from the results dictionary
        ground_truth_df = extract_ground_truth_df(
            ground_truth_results=ground_truth_results, ground_truth_model=ground_truth_model
        )

        models.append(ground_truth_model)
    elif ground_truth_source == "file" and ground_truth_file:
        ground_truth_df = load_ground_truth_file(filepath=ground_truth_file)

    # Select the first two models as primary for visualization (maintaining backward compatibility)
    primary_models = models[:2]

    visualization = evaluate_models(
        model_results=model_results,
        ground_truth_df=ground_truth_df,
        primary_models=primary_models,
    )

    # Save OCR results if requested
    if save_ocr_results_data or save_ground_truth_data:
        save_ocr_results(
            ocr_results=model_results,
            ground_truth_results=ground_truth_df,
            model_names=models,
            output_dir=ocr_results_output_dir,
            ground_truth_output_dir=ground_truth_output_dir,
            save_ground_truth=save_ground_truth_data,
        )

    # Save HTML visualization if requested
    if save_visualization_data:
        save_visualization(visualization, output_dir=visualization_output_dir)


def run_ocr_pipeline(config: Dict[str, Any]) -> None:
    """Run the OCR comparison pipeline from a configuration dictionary.

    Args:
        config: Dictionary containing configuration

    Returns:
        None
    """
    models = config["models"].get("models")
    if not models:
        models = [
            config["models"].get("model1", "llama3.2-vision:11b"),
            config["models"].get("model2", "pixtral-12b-2409"),
        ]

    ocr_comparison_pipeline(
        image_paths=config["input"].get("image_paths"),
        image_folder=config["input"].get("image_folder"),
        custom_prompt=config["models"].get("custom_prompt"),
        models=models,
        ground_truth_model=config["models"].get("ground_truth_model", "gpt-4o-mini"),
        ground_truth_source=config["ground_truth"].get("source", "none"),
        ground_truth_file=config["ground_truth"].get("file"),
        save_ground_truth_data=config["output"]["ground_truth"].get("save", False),
        ground_truth_output_dir=config["output"]["ground_truth"].get("directory", "ocr_results"),
        save_ocr_results_data=config["output"]["ocr_results"].get("save", False),
        ocr_results_output_dir=config["output"]["ocr_results"].get("directory", "ocr_results"),
        save_visualization_data=config["output"]["visualization"].get("save", False),
        visualization_output_dir=config["output"]["visualization"].get(
            "directory", "visualizations"
        ),
    )
