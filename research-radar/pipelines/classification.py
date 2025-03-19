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
Pipeline for article classification and dataset processing.
"""

from typing import Dict, Optional

from steps import (
    classify_articles,
    load_classification_dataset,
    merge_classifications,
    save_classifications,
)
from utils import (
    calculate_and_save_metrics_from_json,
    get_hf_token,
    load_config,
    logger,
)
from zenml import pipeline


@pipeline(enable_cache=False)
def classification_pipeline(config: Optional[Dict] = None):
    """
    Pipeline for article classification and dataset processing.

    Args:
        config: Pipeline configuration from base_config.yaml
                If None, will load from default settings

    Outputs:
        augmentation: Saves to classification_results/augmentation/
        evaluation: Saves to classification_results/for_evaluation/
    """
    config = config or load_config()

    hf_token = get_hf_token()

    pipeline_config = config.steps.classify
    classification_type = pipeline_config.classification_type

    logger.log_classification_type(classification_type)

    dataset_path = (
        config.datasets.unclassified
        if classification_type == "augmentation"
        else config.datasets.composite
    )

    articles = load_classification_dataset(dataset_path)

    classifications = classify_articles(
        articles=articles,
        hf_token=hf_token,
        model_id=config.model_repo_ids.deepseek,
        inference_params=pipeline_config.inference_params,
        classification_type=classification_type,
        batch_config=pipeline_config.batch_processing,
        parallel_config=pipeline_config.parallel_processing,
        checkpoint_config=pipeline_config.checkpoint,
    )

    results_path = save_classifications(
        classifications=classifications,
        classification_type=classification_type,
        model_id=config.model_repo_ids.deepseek,
        inference_params=pipeline_config.inference_params,
        batch_config=pipeline_config.batch_processing,
        checkpoint_config=pipeline_config.checkpoint,
    )

    if classification_type == "evaluation":
        base_dataset_path = config.datasets.composite
        calculate_and_save_metrics_from_json(
            results_path=str(results_path),
            base_dataset_path=base_dataset_path,
        )

    if classification_type == "augmentation":
        merge_classifications(
            results_path=results_path,
            training_dataset_path=config.datasets.composite,
            augmented_dataset_path=config.datasets.augmented,
            source_dataset_path=config.datasets.unclassified,
        )
