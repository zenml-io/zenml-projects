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

import os
from typing import Dict

from schemas import zenml_project
from steps import (
    data_preprocessor,
    data_splitter,
    finetune_modernbert,
    load_training_dataset,
    save_model_local,
    save_test_set,
)
from utils import logger
from utils.remote_setup import determine_if_remote
from zenml import pipeline


@pipeline(model=zenml_project)
def training_pipeline(config: Dict | None = None, save_to_disk: bool = False):
    """ModernBERT fine-tuning pipeline for article classification.

    - Supports local or remote execution via config file selection
    - Comprehensive model evaluation and metadata tracking

    Args:
        config: Pipeline configuration containing training parameters
        save_to_disk: Whether to save the test set to disk

    Dataset Selection Workflow:
    - Prioritizes the augmented dataset (created by the classification pipeline)
    - Falls back to the composite dataset if augmented doesn't exist
    - You can run the classification pipeline in augmentation mode to create the augmented dataset

    Pipeline steps:
    1. Load dataset (augmented or composite)
    2. Preprocess text and labels
    3. Split data (train/val/test)
    4. Fine-tune ModernBERT (locally or on remote hardware)
    5. Export model artifacts
    """
    if not config:
        raise ValueError("Config must be provided")

    remote_execution = determine_if_remote(config)

    datasets = config.get("datasets", {})
    augmented_dataset_path = datasets.get("augmented")
    composite_dataset_path = datasets.get("composite")

    dataset_path = None
    if augmented_dataset_path and os.path.exists(augmented_dataset_path):
        dataset_path = augmented_dataset_path
        logger.info(f"Using augmented dataset: {augmented_dataset_path}")
    elif composite_dataset_path and os.path.exists(composite_dataset_path):
        dataset_path = composite_dataset_path
        logger.info(f"Using composite dataset: {composite_dataset_path}")
    else:
        raise ValueError(
            "No valid dataset found. Check dataset paths in config."
        )

    dataset = load_training_dataset(dataset_path)

    processed_dataset = data_preprocessor(dataset)

    train_set, validation_set, test_set = data_splitter(
        dataset=processed_dataset,
        test_size=config["steps"]["data_split"]["test_size"],
        validation_size=config["steps"]["data_split"]["validation_size"],
    )

    # Save the test set for later evaluation (optional)
    if save_to_disk:
        artifact_path = config["steps"]["compare"]["dataset"]["path"]
        logger.info(f"Saving test set to {artifact_path}")
        save_test_set(
            test_set=test_set,
            artifact_path=artifact_path,
        )

    training_params = config["steps"]["finetune_modernbert"][
        "parameters"
    ].copy()

    base_model_id = config.get("parameters", {}).get(
        "base_model_id"
    ) or config.get("model_repo_ids", {}).get("modernbert_base_model")
    if not base_model_id:
        raise ValueError("Base model ID not found in config")

    model, tokenizer = finetune_modernbert(
        train_set=train_set,
        validation_set=validation_set,
        test_set=test_set,
        training_params=training_params,
        project=config.get("project", {}),
        base_model=base_model_id,
        remote_execution=remote_execution,
    )

    output_paths = config.get("outputs", {})
    model_dir = output_paths.get("ft_model", "models/ft_model")
    tokenizer_dir = output_paths.get("ft_tokenizer", "models/ft_tokenizer")

    save_model_local(
        model=model,
        tokenizer=tokenizer,
        model_dir=model_dir,
        tokenizer_dir=tokenizer_dir,
    )
