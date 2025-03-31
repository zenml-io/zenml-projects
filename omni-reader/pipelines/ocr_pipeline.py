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

import polars as pl
from zenml import pipeline
from zenml.logger import get_logger

from steps import (
    evaluate_models,
    load_ground_truth_file,
    load_images,
    run_ocr,
    save_ocr_results,
    save_visualization,
)

logger = get_logger(__name__)


@pipeline()
def ocr_comparison_pipeline(
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    model1: str = "ollama/gemma3:27b",
    model2: str = "pixtral-12b-2409",
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
    """Run OCR comparison pipeline between two configurable models.

    Args:
        image_paths: Optional list of specific image paths to process
        image_folder: Optional folder to search for images
        custom_prompt: Optional custom prompt to use for both models
        model1: Name of the first model to use (default: ollama/gemma3:27b)
        model2: Name of the second model to use (default: pixtral-12b-2409)
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
    model_names = []

    model1_results = run_ocr(images=images, model_name=model1, custom_prompt=custom_prompt)
    model_names.append(model1)

    model2_results = run_ocr(images=images, model_name=model2, custom_prompt=custom_prompt)
    model_names.append(model2)

    # Handle ground truth based on the selected source
    ground_truth = None
    openai_results = None

    if ground_truth_source == "openai":
        openai_results = run_ocr(
            images=images, model_name=ground_truth_model, custom_prompt=custom_prompt
        )
        ground_truth = openai_results
        model_names.append(ground_truth_model)
    elif ground_truth_source == "file" and ground_truth_file:
        ground_truth = load_ground_truth_file(filepath=ground_truth_file)

    # Evaluate models
    visualization = evaluate_models(
        model1_df=model1_results,
        model2_df=model2_results,
        ground_truth_df=ground_truth,
        model1_name=model1,
        model2_name=model2,
    )

    # Save OCR results if requested
    if save_ocr_results_data or save_ground_truth_data:
        save_ocr_results(
            model1_results=model1_results,
            model2_results=model2_results,
            ground_truth_results=ground_truth,
            model_names=model_names,
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
    ocr_comparison_pipeline(
        image_paths=config["input"].get("image_paths"),
        image_folder=config["input"].get("image_folder"),
        custom_prompt=config["models"].get("custom_prompt"),
        model1=config["models"].get("model1", "ollama/gemma3:27b"),
        model2=config["models"].get("model2", "pixtral-12b-2409"),
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
