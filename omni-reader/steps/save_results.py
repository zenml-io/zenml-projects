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
"""Steps for saving OCR data."""

import os
from datetime import datetime
from typing import Dict, List, Optional

import polars as pl
from zenml import log_metadata, step
from zenml.logger import get_logger
from zenml.types import HTMLString

from utils import save_ocr_data_to_json

logger = get_logger(__name__)


@step(enable_cache=False)
def save_ocr_results(
    ocr_results: Dict[str, pl.DataFrame] = None,
    ground_truth_results: Optional[pl.DataFrame] = None,
    model_names: List[str] = None,
    output_dir: str = "ocr_results",
    ground_truth_output_dir: str = "ocr_results",
    save_ground_truth: bool = False,
    primary_models: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Save OCR results from multiple models.

    Args:
        ocr_results: Dictionary mapping model names to their DataFrame results
        ground_truth_results: Ground truth model OCR results
        model_names: List of model names that were run
        output_dir: Base directory to save OCR results
        ground_truth_output_dir: Directory to save ground truth OCR results
        save_ground_truth: Whether to save ground truth results
        primary_models: Optional list of primary models to focus on

    Returns:
        Dictionary of model names to file paths
    """
    saved_paths = {}
    metadata_dict = {}

    results_mapping = []

    # Get primary models if provided
    if not primary_models and model_names:
        primary_models = model_names[: min(2, len(model_names))]

    # Process all model results from the dictionary
    if ocr_results and model_names:
        for model_name in model_names:
            # Only process models that exist in the results
            if model_name in ocr_results:
                model_prefix = model_name.split("/")[-1].split(":")[0].lower()
                results_mapping.append(
                    {
                        "model_name": model_name,
                        "data": ocr_results[model_name],
                        "subdir": os.path.join(output_dir, model_prefix),
                        "prefix": model_prefix,
                    }
                )

    # Process ground truth results
    if ground_truth_results is not None and save_ground_truth:
        # Handle ground truth as a dictionary (like the model results)
        if isinstance(ground_truth_results, dict) and len(ground_truth_results) > 0:
            # Get the first key as the ground truth model name
            gt_model_name = list(ground_truth_results.keys())[0]
            gt_prefix = "gt_" + gt_model_name.split("/")[-1].split(":")[0].lower()
            results_mapping.append(
                {
                    "model_name": gt_model_name,
                    "data": ground_truth_results[gt_model_name],
                    "subdir": os.path.join(ground_truth_output_dir, "ground_truth"),
                    "prefix": gt_prefix,
                }
            )
        else:
            # Handle direct DataFrame ground truth (old format)
            gt_prefix = "ground_truth"
            results_mapping.append(
                {
                    "model_name": "ground_truth",
                    "data": ground_truth_results,
                    "subdir": os.path.join(ground_truth_output_dir, "ground_truth"),
                    "prefix": gt_prefix,
                }
            )

    # Process each model's results
    for result_info in results_mapping:
        model_name = result_info["model_name"]
        df = result_info["data"]
        subdir = result_info["subdir"]
        prefix = result_info["prefix"]

        filepath = save_ocr_data_to_json(data=df, output_dir=subdir, prefix=prefix)

        saved_paths[model_name] = filepath
        metadata_dict[f"{prefix}_path"] = filepath
        metadata_dict[f"{prefix}_count"] = len(df)

        logger.info(f"{model_name} results saved to: {filepath}")

    # Log metadata
    log_metadata(metadata={"ocr_results_saved": metadata_dict})

    return saved_paths


@step
def save_visualization(
    visualization: HTMLString,
    output_dir: str = "visualizations",
) -> str:
    """Save HTML visualization to a file.

    Args:
        visualization: HTML visualization content
        output_dir: Directory to save visualization

    Returns:
        Path to the saved visualization file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_visualization_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(str(visualization))

    logger.info(f"Visualization saved to: {filepath}")

    log_metadata(
        metadata={
            "visualization_saved": {
                "path": filepath,
            }
        }
    )

    return filepath
