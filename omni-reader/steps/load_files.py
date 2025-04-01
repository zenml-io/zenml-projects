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
"""This module contains the image loader step."""

import glob
import os
from typing import Dict, List, Optional

import polars as pl
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def load_images(
    image_paths: Optional[List[str]] = None,
    image_folder: Optional[str] = None,
) -> List[str]:
    """Load images for OCR processing.

    This step loads images from specified paths or by searching for
    patterns in a given folder.

    Args:
        image_paths: Optional list of specific image paths to load
        image_folder: Optional folder to search for images.

    Returns:
        List of validated image file paths
    """
    all_images = []

    if image_paths:
        all_images.extend(image_paths)
        logger.info(f"Added {len(image_paths)} directly specified images")

    if image_folder:
        patterns_to_use = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif"]

        for pattern in patterns_to_use:
            full_pattern = os.path.join(image_folder, pattern)
            matching_files = glob.glob(full_pattern)
            if matching_files:
                all_images.extend(matching_files)
                logger.info(f"Found {len(matching_files)} images matching pattern {pattern}")

    # Validate image paths
    valid_images = []
    for path in all_images:
        if os.path.isfile(path):
            valid_images.append(path)
        else:
            logger.warning(f"Image not found: {path}")

    # Log metadata about the loaded images
    image_names = [os.path.basename(path) for path in valid_images]
    image_extensions = [os.path.splitext(path)[1].lower() for path in valid_images]

    extension_counts = {}
    for ext in image_extensions:
        if ext in extension_counts:
            extension_counts[ext] += 1
        else:
            extension_counts[ext] = 1

    log_metadata(
        metadata={
            "loaded_images": {
                "total_count": len(valid_images),
                "extensions": extension_counts,
                "image_names": image_names,
            }
        }
    )

    logger.info(f"Successfully loaded {len(valid_images)} valid images")

    return valid_images


@step
def load_ground_truth_file(
    filepath: str,
) -> Annotated[pl.DataFrame, "ground_truth"]:
    """Load ground truth data from a JSON file.

    Args:
        filepath: Path to the ground truth file

    Returns:
        pl.DataFrame containing ground truth results
    """
    from utils.io_utils import load_ocr_data_from_json

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")

    df = load_ocr_data_from_json(filepath)

    log_metadata(metadata={"ground_truth_loaded": {"path": filepath, "count": len(df)}})

    return df
