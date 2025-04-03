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
"""This module contains the ground truth and OCR results loader steps."""

import glob
import json
import os
from typing import Dict, List, Optional

import polars as pl
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.logger import get_logger

from utils.model_configs import get_model_prefix

logger = get_logger(__name__)


@step()
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
        patterns_to_use = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.tiff"]

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


@step(enable_cache=False)
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


@step(enable_cache=False)
def load_ground_truth_texts(
    model_results: Dict[str, pl.DataFrame],
    ground_truth_folder: Optional[str] = None,
    ground_truth_files: Optional[List[str]] = None,
) -> Annotated[pl.DataFrame, "ground_truth"]:
    """Load ground truth texts using image names found in model results."""
    if not ground_truth_folder and not ground_truth_files:
        raise ValueError("Either ground_truth_folder or ground_truth_files must be provided")

    # Grab image names from any model result
    sample_model_df = next(iter(model_results.values()))
    image_names = sample_model_df.select("image_name").to_series().to_list()

    gt_path_map = {}

    if ground_truth_folder:
        for f in os.listdir(ground_truth_folder):
            if f.endswith(".txt"):
                base = os.path.splitext(f)[0]
                gt_path_map[base] = os.path.join(ground_truth_folder, f)
    elif ground_truth_files:
        for path in ground_truth_files:
            base = os.path.splitext(os.path.basename(path))[0]
            gt_path_map[base] = path

    data = []
    missing = []

    for i, img_name in enumerate(image_names):
        base_name = os.path.splitext(img_name)[0]
        gt_path = gt_path_map.get(base_name)

        if not gt_path or not os.path.exists(gt_path):
            missing.append(img_name)
            continue

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            data.append(
                {
                    "id": i,
                    "image_name": img_name,
                    "raw_text": text,
                    "processing_time": 0,
                    "confidence": 1.0,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to read ground truth for {img_name}: {e}")

    if missing:
        logger.warning(
            f"Missing ground truth files for {len(missing)} images: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    if not data:
        raise ValueError("No ground truth files could be loaded.")

    return pl.DataFrame(data)


@step(enable_cache=False)
def load_ocr_results(
    model_names: List[str],
    results_dir: str = "ocr_results",
    result_files: Optional[List[str]] = None,
) -> Dict[str, pl.DataFrame]:
    """Load OCR results from previously saved JSON files."""
    results = {}
    model_to_prefix = {model: get_model_prefix(model) for model in model_names}

    if result_files:
        for file_path in result_files:
            if not os.path.exists(file_path):
                logger.warning(f"Result file not found: {file_path}")
                continue

            file_name = os.path.basename(file_path)
            for model, prefix in model_to_prefix.items():
                # Check for exact prefix match at start of filename
                if file_name.startswith(f"{prefix}_"):
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if "ocr_data" in data:
                                results[model] = pl.DataFrame(data["ocr_data"])
                            else:
                                results[model] = pl.DataFrame(data)
                        break
                    except Exception as e:
                        logger.error(f"Error loading {model} results: {str(e)}")
    else:
        for model, prefix in model_to_prefix.items():
            model_dir = os.path.join(results_dir, prefix)
            if not os.path.exists(model_dir):
                logger.warning(f"No results directory found for model: {model}")
                continue

            # Find files matching the exact prefix pattern
            json_files = glob.glob(os.path.join(model_dir, f"{prefix}_*.json"))
            if not json_files:
                logger.warning(f"No result files found for model: {model}")
                continue

            latest_file = sorted(json_files, key=os.path.getmtime, reverse=True)[0]
            logger.info(f"Loading results for {model} from {latest_file}")

            try:
                with open(latest_file, "r") as f:
                    data = json.load(f)
                    if "ocr_data" in data:
                        results[model] = pl.DataFrame(data["ocr_data"])
                    else:
                        results[model] = pl.DataFrame(data)
            except Exception as e:
                logger.error(f"Error loading results for {model}: {str(e)}")

    if not results:
        raise ValueError("No model results could be loaded. Run the batch pipeline first.")

    return results
