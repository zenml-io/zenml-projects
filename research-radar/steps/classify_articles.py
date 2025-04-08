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
ZenML step for classifying articles using HuggingFace models.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient
from schemas import (
    BatchProcessingConfig,
    CheckpointConfig,
    InferenceParamsConfig,
    InputArticle,
    ParallelProcessingConfig,
)
from utils import (
    logger,
    process_articles_parallel,
    process_articles_sequential,
)
from utils.checkpoint import find_checkpoint_file, load_checkpoint_data
from zenml.steps import step


@step(enable_cache=False)
def classify_articles(
    articles: List[InputArticle],
    hf_token: str,
    model_id: str,
    inference_params: InferenceParamsConfig,
    classification_type: str,
    batch_config: Optional[BatchProcessingConfig] = None,
    parallel_config: Optional[ParallelProcessingConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
) -> List[Dict]:
    """
    Batch classifies articles using HuggingFace inference API.
    If batch_config is not provided, all articles will be processed.

    Args:
        articles: List of articles to classify
        hf_token: HuggingFace API token
        model_id: DeepSeek model identifier
        inference_params: Model inference parameters
        classification_type: Type of classification ('evaluation' or 'augmentation')
        batch_config: Batch processing parameters (optional)
        parallel_config: Parallel processing parameters (optional)
        checkpoint_config: Checkpoint configuration (optional)

    Returns:
        List of classification results with metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_params_dict = inference_params.model_dump()

    output_dir = os.path.join(
        f"classification_results/for_{classification_type}", timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    batch_articles = (
        articles[
            batch_config.batch_start : batch_config.batch_start
            + batch_config.batch_size
        ]
        if batch_config
        else articles
    )

    remaining_articles = batch_articles
    if (
        checkpoint_config
        and checkpoint_config.enabled
        and checkpoint_config.run_id
    ):
        checkpoint_path = find_checkpoint_file(
            classification_type, checkpoint_config.run_id
        )
        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_data, completed_article_urls, checkpoint_progress = (
                load_checkpoint_data(checkpoint_path)
            )

            remaining_articles = [
                article
                for article in batch_articles
                if str(article.meta.url) not in completed_article_urls
            ]

            logger.log_checkpoint(
                f"Resuming from checkpoint with run_id: {checkpoint_config.run_id}"
            )
            logger.log_checkpoint(
                f"Found {len(completed_article_urls)} processed articles, {len(remaining_articles)} remaining"  # noqa: E501
            )

            output_dir = checkpoint_dir

    client = InferenceClient(token=hf_token)
    logger.log_classification_config(
        remaining_articles=remaining_articles,
        parallel_config=parallel_config,
        checkpoint_config=checkpoint_config,
        batch_config=batch_config,
        total_articles=len(batch_articles),
    )

    parallel_workers = (
        parallel_config.workers
        if parallel_config and parallel_config.enabled
        else 1
    )
    if parallel_workers < 1:
        logger.log_warning(
            f"Invalid parallel_workers value: {parallel_workers}, using 1 instead"
        )
        parallel_workers = 1

    if (
        parallel_config
        and parallel_config.enabled
        and len(remaining_articles) > 1
    ):
        results = process_articles_parallel(
            remaining_articles=remaining_articles,
            batch_articles=batch_articles,
            client=client,
            model_id=model_id,
            inference_params_dict=inference_params_dict,
            parallel_workers=parallel_workers,
            checkpoint_config=checkpoint_config,
            classification_type=classification_type,
            output_dir=output_dir,
            batch_config=batch_config,
        )
    else:
        results = process_articles_sequential(
            remaining_articles=remaining_articles,
            batch_articles=batch_articles,
            client=client,
            model_id=model_id,
            inference_params_dict=inference_params_dict,
            checkpoint_config=checkpoint_config,
            classification_type=classification_type,
            output_dir=output_dir,
            batch_config=batch_config,
        )

    filtered_results = [r for r in results if r is not None]

    logger.log_checkpoint(
        f"Classification complete. Processed {len(filtered_results)} articles."
    )
    if checkpoint_config and checkpoint_config.enabled:
        logger.log_checkpoint(f"Final run_id: {checkpoint_config.run_id}")

    return filtered_results
