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
"""Utilities for saving and loading data."""

import json
import os
from datetime import datetime
from typing import List

import polars as pl


def save_ocr_data_to_json(
    data: pl.DataFrame,
    output_dir: str,
    prefix: str,
) -> str:
    """Save OCR data (ground truth or model results) to a JSON file.

    Args:
        data: DataFrame containing OCR results
        output_dir: Directory to save the data
        prefix: Prefix for the output filename

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert to list of dictionaries
    json_data = data.to_dicts()

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Save to JSON
    with open(filepath, "w") as f:
        json.dump(
            {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "image_count": len(json_data),
                    "format_version": "1.0",
                },
                "ocr_data": json_data,
            },
            f,
            indent=2,
        )

    return filepath


def load_ocr_data_from_json(filepath: str) -> pl.DataFrame:
    """Load OCR data from a JSON file.

    Args:
        filepath: Path to the JSON file containing OCR data

    Returns:
        DataFrame containing OCR data
    """
    # Load JSON data
    with open(filepath, "r") as f:
        data = json.load(f)

    # Extract data - handle both new and legacy formats
    if "ocr_data" in data:
        ocr_data = data["ocr_data"]
    elif "ground_truth_data" in data:
        ocr_data = data["ground_truth_data"]
    else:
        ocr_data = data  # Assume the file contains just the data

    # Convert to DataFrame
    return pl.DataFrame(ocr_data)


def load_ground_truth_from_json(filepath: str) -> pl.DataFrame:
    """Load ground truth data from a JSON file.

    Args:
        filepath: Path to the ground truth JSON file

    Returns:
        DataFrame containing ground truth data
    """
    return load_ocr_data_from_json(filepath)


def list_available_ground_truth_files(
    directory: str = "ocr_results/ground_truth", pattern: str = "gt_*.json"
) -> List[str]:
    """List available ground truth files.

    Args:
        directory: Directory containing ground truth files
        pattern: Glob pattern to match files

    Returns:
        List of paths to ground truth files
    """
    import glob

    # Create path pattern
    path_pattern = os.path.join(directory, pattern)

    # Find matching files
    files = glob.glob(path_pattern)

    return sorted(files, reverse=True)  # Sort by newest first
