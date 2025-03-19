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
#

"""
Utility functions for checkpointing the classification process.
"""

import json
import os
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from schemas import CheckpointConfig, InputArticle
from utils import logger


def find_checkpoint_file(classification_type: str, run_id: str) -> Optional[str]:
    """
    Find a checkpoint file with the specified run_id.

    Args:
        classification_type: Type of classification ('evaluation' or 'augmentation')
        run_id: Unique identifier for the run

    Returns:
        Path to the checkpoint file if found, None otherwise
    """
    base_dir = f"classification_results/for_{classification_type}"
    pattern = f"{base_dir}/**/results.json"
    checkpoint_files = glob(pattern, recursive=True)

    for file_path in checkpoint_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if data.get("run_id") == run_id:
                return file_path
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.error(f"Error reading checkpoint file {file_path}: {e}")

    return None


def load_checkpoint_data(checkpoint_path: str) -> Tuple[Dict[str, Any], List[str], int]:
    """
    Load data from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple containing (checkpoint_data, completed_article_urls, checkpoint_progress)
    """
    with open(checkpoint_path, "r") as f:
        checkpoint_data = json.load(f)

    # Get URLs of articles that have already been processed
    completed_article_urls = []
    for idx, url in checkpoint_data.get("results", {}).get("meta_url", {}).items():
        completed_article_urls.append(url)

    # Get checkpoint progress
    checkpoint_progress = checkpoint_data.get("metadata", {}).get("checkpoint_progress", 0)

    return checkpoint_data, completed_article_urls, checkpoint_progress


def save_checkpoint(
    results: List[Dict],
    batch_articles: List[InputArticle],
    classification_type: str,
    model_id: str,
    inference_params_dict: Dict,
    checkpoint_config: CheckpointConfig,
    batch_config: Any,
    output_dir: str,
    progress: int,
) -> None:
    """
    Save a checkpoint of the current processing state.

    Args:
        results: Current classification results
        batch_articles: All articles in the batch
        classification_type: Type of classification
        model_id: Model identifier
        inference_params_dict: Inference parameters
        checkpoint_config: Checkpoint configuration
        batch_config: Batch processing configuration
        output_dir: Directory to save checkpoint to
        progress: Number of articles processed so far
    """
    # Determine batch information
    if batch_config is not None:
        batch_start = batch_config.batch_start + 1
        batch_size = batch_config.batch_size
        batch_end = batch_start + batch_size - 1
    else:
        batch_start = 1
        batch_end = len(batch_articles)

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"classification_results/for_{classification_type}"
        output_dir = os.path.join(base_dir, timestamp)

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "results.json")

    filtered_results = [r for r in results if r is not None]

    # to avoid circular imports
    from utils.classification_helpers import prepare_classification_json

    # Prepare and save JSON data with checkpoint information
    json_dict = prepare_classification_json(
        classifications=filtered_results,
        batch_start=batch_start,
        batch_end=batch_end,
        inference_params_dict=inference_params_dict,
        model_id=model_id,
        run_id=checkpoint_config.run_id,
        is_checkpoint=True,
        checkpoint_progress=progress,
    )

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)

    logger.log_checkpoint(
        f"Checkpoint saved at {checkpoint_path} (progress: {progress}/{len(batch_articles)} articles)"
    )
