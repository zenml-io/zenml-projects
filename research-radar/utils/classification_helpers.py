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
Utility functions for article classification with HuggingFace models.
"""

import concurrent.futures
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient
from schemas import CheckpointConfig, ClassificationOutput, InputArticle
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from utils import (
    format_prompt_for_deepseek,
    logger,
    try_extract_json_from_text,
)
from utils.checkpoint import save_checkpoint


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2),
    retry=retry_if_exception_type((Exception)),
    reraise=True,
)
def classify_single_article(
    client: InferenceClient,
    article: InputArticle,
    model_id: str,
    inference_params: Dict,
) -> Dict:
    """
    Classifies a single article using the HuggingFace inference API.

    Args:
        client: HuggingFace inference client instance
        article: Article to be classified
        model_id: DeepSeek model identifier
        inference_params: Model inference parameters

    Returns:
        Dict containing classification results and article metadata

    Raises:
        ValueError: If API response is empty or invalid JSON
    """
    max_tokens = (
        inference_params["max_sequence_length"]
        - inference_params["max_new_tokens"]
    )
    prompt = format_prompt_for_deepseek(article.text, max_tokens)

    response = client.text_generation(
        prompt,
        model=model_id,
        temperature=inference_params["temperature"],
        top_p=inference_params["top_p"],
        top_k=inference_params["top_k"],
        return_full_text=False,
    )

    if not response or not response.strip():
        raise ValueError("Empty response from API")

    _, json_content = try_extract_json_from_text(response)
    if json_content is None:
        raise ValueError("No JSON found in model output")

    validated = ClassificationOutput(**json_content)
    return {**validated.model_dump(), "meta": article.meta.model_dump()}


def classify_article_safely(args: Tuple) -> Tuple[int, Dict]:
    """
    Helper function for parallel processing with error handling.

    Args:
        args: Tuple containing (index, article, client, model_id, inference_params)

    Returns:
        Tuple of (index, result_dict) where result_dict contains classification or error info
    """
    index, article, client, model_id, inference_params = args

    try:
        result = classify_single_article(
            client=client,
            article=article,
            model_id=model_id,
            inference_params=inference_params,
        )
        return index, result
    except Exception as e:
        error_msg = f"Classification failed for {article.meta.url}: {str(e)}"
        logger.error(error_msg)
        # Return a standardized error result
        return index, {
            "is_accepted": False,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}",
            "meta": article.meta.model_dump(),
        }


def prepare_classification_json(
    classifications: List[Dict],
    batch_start: int,
    batch_end: int,
    inference_params_dict: Dict,
    model_id: str,
    run_id: str = None,
    is_checkpoint: bool = False,
    checkpoint_progress: int = None,
) -> Dict:
    """
    Prepare a structured JSON with classification results.

    Args:
        classifications: List of classification results
        batch_start: Start index of batch (1-based)
        batch_end: End index of batch (1-based)
        inference_params_dict: Dictionary of inference parameters
        model_id: Model identifier used for classification
        run_id: Unique identifier for this classification run
        is_checkpoint: Whether this is a checkpoint file
        checkpoint_progress: Number of articles processed so far

    Returns:
        Dictionary with structured results
    """
    result_dict = {
        "run_id": run_id or str(uuid.uuid4())[:8],
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "inference_params": inference_params_dict,
            "batch_info": {
                "start": batch_start,
                "end": batch_end,
                "size": len(classifications),
            },
            "is_checkpoint": is_checkpoint,
        },
        "results": {
            "is_accepted": {},
            "confidence": {},
            "reason": {},
            "meta_url": {},
        },
    }

    if checkpoint_progress is not None:
        result_dict["metadata"]["checkpoint_progress"] = checkpoint_progress

    # Process classifications with 1-based indexing
    for i, classification in enumerate(classifications):
        idx = str(i + 1)
        result_dict["results"]["is_accepted"][idx] = classification.get(
            "is_accepted", False
        )
        result_dict["results"]["confidence"][idx] = classification.get(
            "confidence", 0.0
        )
        result_dict["results"]["reason"][idx] = classification.get(
            "reason", ""
        )

        meta = classification.get("meta", {})

        if "url" in meta:
            if "meta_url" not in result_dict["results"]:
                result_dict["results"]["meta_url"] = {}
            result_dict["results"]["meta_url"][idx] = str(meta.get("url", ""))
        if "published_date" in meta:
            if "meta_published_date" not in result_dict["results"]:
                result_dict["results"]["meta_published_date"] = {}
            result_dict["results"]["meta_published_date"][idx] = str(
                meta.get("published_date", "")
            )
        if "title" in meta:
            if "meta_title" not in result_dict["results"]:
                result_dict["results"]["meta_title"] = {}
            result_dict["results"]["meta_title"][idx] = meta.get("title", "")
        if "author" in meta:
            if "meta_author" not in result_dict["results"]:
                result_dict["results"]["meta_author"] = {}
            result_dict["results"]["meta_author"][idx] = meta.get("author", "")

    return result_dict


def process_articles_parallel(
    remaining_articles: List[InputArticle],
    batch_articles: List[InputArticle],
    client: InferenceClient,
    model_id: str,
    inference_params_dict: Dict,
    parallel_workers: int,
    checkpoint_config: Optional[CheckpointConfig] = None,
    classification_type: str = "evaluation",
    output_dir: str = None,
    batch_config: Optional[Any] = None,
) -> List[Optional[Dict]]:
    """
    Process articles in parallel using ThreadPoolExecutor.

    Args:
        remaining_articles: List of articles to process
        batch_articles: Full list of articles in the batch
        client: HuggingFace inference client
        model_id: Model identifier
        inference_params_dict: Inference parameters
        parallel_workers: Number of parallel workers
        checkpoint_config: Checkpoint configuration
        classification_type: Type of classification
        output_dir: Directory to save checkpoint to

    Returns:
        Tuple containing (results, completed_indices)
    """

    results = [None] * len(batch_articles)
    completed_indices = set()
    checkpoint_enabled = checkpoint_config and checkpoint_config.enabled
    checkpoint_frequency = (
        checkpoint_config.frequency if checkpoint_enabled else 0
    )

    # Generate a run_id if not provided
    if checkpoint_enabled and not checkpoint_config.run_id:
        checkpoint_config.run_id = str(uuid.uuid4())[:8]
        logger.log_checkpoint(f"Generated run_id: {checkpoint_config.run_id}")

    # Map remaining articles to their original indices
    process_args = []
    article_to_original_idx = {}

    for i, article in enumerate(remaining_articles):
        original_idx = next(
            idx
            for idx, a in enumerate(batch_articles)
            if a.meta.url == article.meta.url
        )
        process_args.append(
            (i, article, client, model_id, inference_params_dict)
        )
        article_to_original_idx[i] = original_idx

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=parallel_workers
    ) as executor:
        future_to_idx = {
            executor.submit(classify_article_safely, args): args[0]
            for args in process_args
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing articles",
            unit="article",
        ):
            try:
                temp_idx, result = future.result()
                original_idx = article_to_original_idx[temp_idx]
                results[original_idx] = result
                completed_indices.add(original_idx)

                article = batch_articles[original_idx]
                logger.log_classification(
                    result["is_accepted"],
                    result["reason"],
                    article.meta.title,
                    article.meta.url,
                )

            except Exception as exc:
                temp_idx = future_to_idx[future]
                original_idx = article_to_original_idx[temp_idx]
                logger.error(
                    f"Article #{original_idx} generated an exception: {exc}"
                )
                results[original_idx] = {
                    "is_accepted": False,
                    "confidence": 0.0,
                    "reason": f"Unexpected error: {str(exc)}",
                    "meta": batch_articles[original_idx].meta.model_dump(),
                }
                completed_indices.add(original_idx)

            # Save checkpoint if enabled and at checkpoint frequency
            progress = len(completed_indices)
            if (
                checkpoint_enabled
                and progress % checkpoint_frequency == 0
                and progress > 0
            ):
                save_checkpoint(
                    results=results,
                    batch_articles=batch_articles,
                    classification_type=classification_type,
                    model_id=model_id,
                    inference_params_dict=inference_params_dict,
                    checkpoint_config=checkpoint_config,
                    batch_config=batch_config,
                    output_dir=output_dir,
                    progress=progress,
                )

    return results


def process_articles_sequential(
    remaining_articles: List[InputArticle],
    batch_articles: List[InputArticle],
    client: InferenceClient,
    model_id: str,
    inference_params_dict: Dict,
    checkpoint_config: Optional[CheckpointConfig] = None,
    classification_type: str = "evaluation",
    output_dir: str = None,
    batch_config: Optional[Any] = None,
) -> List[Optional[Dict]]:
    """
    Process articles sequentially with a progress bar.

    Args:
        remaining_articles: List of articles to process
        batch_articles: Full list of articles in the batch
        client: HuggingFace inference client
        model_id: Model identifier
        inference_params_dict: Inference parameters
        checkpoint_config: Checkpoint configuration
        classification_type: Type of classification
        output_dir: Directory to save checkpoint to

    Returns:
        Tuple containing (results, completed_indices)
    """
    results = [None] * len(batch_articles)
    completed_indices = set()
    checkpoint_enabled = checkpoint_config and checkpoint_config.enabled
    checkpoint_frequency = (
        checkpoint_config.frequency if checkpoint_enabled else 0
    )

    if checkpoint_enabled and not checkpoint_config.run_id:
        checkpoint_config.run_id = str(uuid.uuid4())[:8]
        logger.log_checkpoint(f"Generated run_id: {checkpoint_config.run_id}")

    for i, article in tqdm(
        enumerate(remaining_articles),
        total=len(remaining_articles),
        desc="Processing articles",
        unit="article",
    ):
        try:
            result = classify_single_article(
                client=client,
                article=article,
                model_id=model_id,
                inference_params=inference_params_dict,
            )
            logger.log_classification(
                result["is_accepted"],
                result["reason"],
                title=batch_articles[i].meta.title,
                url=batch_articles[i].meta.url,
            )
        except Exception as e:
            logger.error(f"Classification failed for {article.meta.url}: {e}")
            result = {
                "is_accepted": False,
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "meta": article.meta.model_dump(),
            }

        # Find original index of this article in batch_articles
        original_idx = next(
            idx
            for idx, a in enumerate(batch_articles)
            if a.meta.url == article.meta.url
        )
        results[original_idx] = result
        completed_indices.add(original_idx)

        # Save checkpoint if enabled and at checkpoint frequency
        progress = len(completed_indices)
        if checkpoint_enabled and progress % checkpoint_frequency == 0:
            save_checkpoint(
                results=results,
                batch_articles=batch_articles,
                classification_type=classification_type,
                model_id=model_id,
                inference_params_dict=inference_params_dict,
                checkpoint_config=checkpoint_config,
                batch_config=batch_config,
                output_dir=output_dir,
                progress=progress,
            )

    return results
