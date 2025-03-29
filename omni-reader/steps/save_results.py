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
from typing import Dict

import polars as pl
from zenml import log_metadata, step
from zenml.logger import get_logger
from zenml.types import HTMLString

from utils.io_utils import save_ocr_data_to_json

logger = get_logger(__name__)


@step
def save_ocr_results(
    results_dict: Dict[str, Dict[str, pl.DataFrame]],
    output_dir: str = "ocr_results",
    ground_truth_output_dir: str = "ocr_results",
    save_ground_truth: bool = False,
) -> Dict[str, str]:
    """Save OCR results from multiple models.

    Args:
        results_dict: Dictionary with model name keys and result dict values
        output_dir: Base directory to save OCR results
        ground_truth_output_dir: Directory to save ground truth OCR results
        save_ground_truth: Whether to save ground truth results
                          (only saves if 'gpt-4o-mini' is in results_dict)

    Returns:
        Dictionary of model names to file paths
    """
    saved_paths = {}
    metadata_dict = {}

    for model_name, result_dict in results_dict.items():
        if model_name == "gpt-4o-mini" and not save_ground_truth:
            continue

        if model_name == "gpt-4o-mini":
            subdir = os.path.join(ground_truth_output_dir, "ground_truth")
            prefix = "gt_openai"
            key_name = "ground_truth_results"
        elif "gemma" in model_name.lower():
            subdir = os.path.join(output_dir, "gemma")
            prefix = "gemma"
            key_name = "gemma_results"
        elif "mistral" in model_name.lower() or "pixtral" in model_name.lower():
            subdir = os.path.join(output_dir, "mistral")
            prefix = "mistral"
            key_name = "mistral_results"
        else:
            # For other models, use model name as subdirectory and key
            model_short_name = model_name.lower().split("/")[-1]
            subdir = os.path.join(output_dir, model_short_name)
            prefix = model_short_name
            key_name = list(result_dict.keys())[0]  # Use the first key in the result dict

        # Save results to file
        file_path = save_ocr_data_to_json(data=result_dict, output_dir=subdir, prefix=prefix, key_name=key_name)

        saved_paths[model_name] = file_path

        # Get count for metadata
        result_df = list(result_dict.values())[0]
        count = len(result_df)

        metadata_dict[f"{prefix}_path"] = file_path
        metadata_dict[f"{prefix}_count"] = count

        logger.info(f"{model_name} results saved to: {file_path}")

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

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_visualization_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    # Write HTML to file
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
