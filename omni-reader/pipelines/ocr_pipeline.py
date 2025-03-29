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
    image_patterns: Optional[List[str]] = None,
    custom_prompt: Optional[str] = None,
    ground_truth_texts: Optional[List[str]] = None,
    ground_truth_source: Literal["openai", "manual", "file", "none"] = "none",
    ground_truth_file: Optional[str] = None,
    save_ground_truth_data: bool = False,
    ground_truth_output_dir: str = "ground_truth",
    save_ocr_results_data: bool = False,
    ocr_results_output_dir: str = "ocr_results",
    save_visualization_data: bool = False,
    visualization_output_dir: str = "visualizations",
) -> None:
    """Run OCR comparison pipeline between Gemma3 and Mistral models.

    Args:
        image_paths: Optional list of specific image paths to process
        image_folder: Optional folder to search for images
        image_patterns: Optional list of glob patterns to use when searching image_folder
        custom_prompt: Optional custom prompt to use for both models
        ground_truth_texts: Optional list of ground truth texts for evaluation (used when ground_truth_source="manual")
        ground_truth_source: Source of ground truth - "openai" to use GPT-4V, "manual" for user-provided texts,
                            "file" to load from a saved JSON file, or "none" to skip ground truth evaluation
        ground_truth_file: Path to ground truth JSON file (used when ground_truth_source="file")
        save_ground_truth_data: Whether to save generated ground truth data for future use
        ground_truth_output_dir: Directory to save ground truth data
        save_ocr_results_data: Whether to save OCR results from Gemma and Mistral
        ocr_results_output_dir: Directory to save OCR results
        save_visualization_data: Whether to save HTML visualization to local file
        visualization_output_dir: Directory to save HTML visualization

    Returns:
        None
    """
    images = load_images(
        image_paths=image_paths,
        image_folder=image_folder,
        image_patterns=image_patterns,
    )

    # Keep track of which models were run
    model_names = []

    # Run models in parallel on all images using the unified OCR step
    gemma_results = run_ocr(images=images, model_name="ollama/gemma3:27b", custom_prompt=custom_prompt)
    model_names.append("ollama/gemma3:27b")

    mistral_results = run_ocr(images=images, model_name="pixtral-12b-2409", custom_prompt=custom_prompt)
    model_names.append("pixtral-12b-2409")

    # Handle ground truth based on the selected source
    ground_truth = None
    openai_results = None

    if ground_truth_source == "openai":
        openai_results = run_ocr(images=images, model_name="gpt-4o-mini", custom_prompt=custom_prompt)
        ground_truth = openai_results
        model_names.append("gpt-4o-mini")

    elif ground_truth_source == "manual" and ground_truth_texts:
        ground_truth_data = []
        for i, (text, image_path) in enumerate(zip(ground_truth_texts, images)):
            ground_truth_data.append(
                {
                    "id": i,
                    "image_name": os.path.basename(image_path),
                    "raw_text": text,
                    "confidence": 1.0,  # Manual ground truth has perfect confidence
                }
            )
        ground_truth_df = pl.DataFrame(ground_truth_data)
        ground_truth = {"ground_truth_results": ground_truth_df}

    elif ground_truth_source == "file" and ground_truth_file:
        ground_truth = load_ground_truth_file(filepath=ground_truth_file)

    # Evaluate models
    visualization = evaluate_models(
        gemma_results=gemma_results,
        mistral_results=mistral_results,
        ground_truth=ground_truth,
    )

    # Save OCR results if requested
    if save_ocr_results_data or save_ground_truth_data:
        save_ocr_results(
            gemma_results=gemma_results,
            mistral_results=mistral_results,
            openai_results=openai_results,
            model_names=model_names,
            output_dir=ocr_results_output_dir,
            ground_truth_output_dir=ground_truth_output_dir,
            save_ground_truth=save_ground_truth_data,
        )

    # Save HTML visualization if requested
    if save_visualization_data:
        save_visualization(
            visualization=visualization,
            output_dir=visualization_output_dir,
        )


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
        image_patterns=config["input"].get("image_patterns"),
        custom_prompt=config["models"].get("custom_prompt"),
        ground_truth_texts=config["ground_truth"].get("texts"),
        ground_truth_source=config["ground_truth"].get("source", "none"),
        ground_truth_file=config["ground_truth"].get("file"),
        save_ground_truth_data=config["output"]["ground_truth"].get("save", False),
        ground_truth_output_dir=config["output"]["ground_truth"].get("directory", "ocr_results"),
        save_ocr_results_data=config["output"]["ocr_results"].get("save", False),
        ocr_results_output_dir=config["output"]["ocr_results"].get("directory", "ocr_results"),
        save_visualization_data=config["output"]["visualization"].get("save", False),
        visualization_output_dir=config["output"]["visualization"].get("directory", "visualizations"),
    )
