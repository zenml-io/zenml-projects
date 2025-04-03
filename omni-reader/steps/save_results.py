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
from utils.model_configs import MODEL_CONFIGS, get_model_info

logger = get_logger(__name__)


@step()
def save_ocr_results(
    ocr_results: Dict[str, pl.DataFrame] = None,
    model_names: List[str] = None,
    output_dir: str = "ocr_results",
) -> Dict[str, str]:
    """Save OCR results from multiple models.

    Args:
        ocr_results: Dictionary mapping model names to their DataFrame results
        model_names: List of model names that were run
        output_dir: Base directory to save OCR results

    Returns:
        Dictionary of model names to file paths
    """
    saved_paths = {}
    metadata_dict = {}

    results_mapping = []

    if not model_names and ocr_results:
        model_names = list(ocr_results.keys())

    if ocr_results and model_names:
        for model_name in model_names:
            if model_name in ocr_results:
                if model_name in MODEL_CONFIGS:
                    model_prefix = MODEL_CONFIGS[model_name].prefix
                else:
                    _, model_prefix = get_model_info(model_name)

                results_mapping.append(
                    {
                        "model_name": model_name,
                        "data": ocr_results[model_name],
                        "subdir": os.path.join(output_dir, model_prefix),
                        "prefix": model_prefix,
                    }
                )

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
