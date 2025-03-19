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
Step for saving classification results with automatic metrics calculation.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from schemas import (
    BatchProcessingConfig,
    CheckpointConfig,
    InferenceParamsConfig,
)
from utils import (
    logger,
    prepare_classification_json,
)
from zenml import step


@step(enable_cache=False)
def save_classifications(
    classifications: List[Dict],
    classification_type: str,
    model_id: str,
    inference_params: InferenceParamsConfig,
    batch_config: Optional[BatchProcessingConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    output_dir: Optional[str] = None,
):
    """
    Save classification results to JSON.

    Args:
        classifications: List of classification results
        classification_type: Type of classification (evaluation or augmentation)
        model_id: Model identifier used for classification
        inference_params: Model inference parameters
        batch_config: Batch processing parameters (optional)
        checkpoint_config: Checkpoint configuration (optional)
        output_dir: Directory to save results to (optional, default is timestamped dir)

    Returns:
        String path to the saved results file
    """
    if batch_config is not None:
        batch_start = batch_config.batch_start + 1
        batch_size = batch_config.batch_size
        batch_end = batch_start + batch_size - 1
    else:
        batch_start = 1
        batch_end = len(classifications)

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"classification_results/for_{classification_type}"
        output_dir = os.path.join(base_dir, timestamp)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "results.json")

    run_id = None
    if checkpoint_config and checkpoint_config.enabled:
        run_id = checkpoint_config.run_id

    json_dict = prepare_classification_json(
        classifications=classifications,
        batch_start=batch_start,
        batch_end=batch_end,
        inference_params_dict=inference_params.model_dump(),
        model_id=model_id,
        run_id=run_id,
        is_checkpoint=False,  # This is the final result, not a checkpoint
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2, ensure_ascii=False)

    logger.log_output_file(output_path, "Classification Results")

    if checkpoint_config and checkpoint_config.enabled:
        logger.log_checkpoint(f"Final results saved with run_id: {run_id}")

    return output_path
