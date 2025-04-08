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
"""Utility functions for OCR operations across different models."""

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import polars as pl
import requests
from dotenv import load_dotenv
from tqdm import tqdm
from zenml import log_metadata
from zenml.logger import get_logger

from utils.encode_image import encode_image
from utils.extract_json import try_extract_json_from_response
from utils.model_configs import MODEL_CONFIGS, ModelConfig
from utils.prompt import ImageDescription, get_prompt

load_dotenv()

logger = get_logger(__name__)


# ============================================================================
# Metadata Logging Functions
# ============================================================================


def log_image_metadata(
    prefix: str,
    index: int,
    image_name: str,
    processing_time: float,
    text_length: int,
    confidence: float,
):
    """Log metadata for a processed image."""
    log_metadata(
        metadata={
            f"{prefix}_image_{index}": {
                "image_name": image_name,
                "processing_time_seconds": processing_time,
                "text_length": text_length,
                "confidence": confidence,
            }
        }
    )


def log_error_metadata(
    prefix: str,
    index: int,
    image_name: str,
    error: str,
):
    """Log error metadata for a failed image processing."""
    log_metadata(
        metadata={
            f"{prefix}_error_image_{index}": {
                "image_name": image_name,
                "error": error,
            }
        }
    )


def log_summary_metadata(
    prefix: str,
    model_name: str,
    images_count: int,
    processing_times: List[float],
    confidence_scores: List[float],
):
    """Log summary metadata for all processed images."""
    avg_time = statistics.mean(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0

    log_metadata(
        metadata={
            f"{prefix}_ocr_summary": {
                "model": model_name,
                "images_processed": images_count,
                "avg_processing_time": avg_time,
                "min_processing_time": min_time,
                "max_processing_time": max_time,
                "avg_confidence": avg_confidence,
                "total_processing_time": sum(processing_times),
            }
        }
    )


# ============================================================================
# Model Processing Functions
# ============================================================================


def process_ollama_based(model_name: str, prompt: str, image_base64: str) -> Dict[str, Any]:
    """Process an image with an Ollama model."""
    BASE_URL = os.getenv("OLLAMA_HOST") or "http://localhost:11434/api/generate"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "images": [image_base64],
    }

    try:
        response = requests.post(
            BASE_URL,
            json=payload,
            timeout=120,  # 2mins, in case of really complex images
        )
        response.raise_for_status()
        res = response.json().get("response", "")
        result_json = try_extract_json_from_response(res)

        return result_json
    except Exception as e:
        logger.error(f"Error processing with Ollama model {model_name}: {str(e)}")
        return {"raw_text": f"Error: {str(e)}", "confidence": 0.0}


def process_openai_based(model_name: str, prompt: str, image_url: str) -> Dict[str, Any]:
    """Process an image with an API-based model (OpenAI)."""
    import instructor
    from openai import OpenAI

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    client = instructor.from_openai(openai_client)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_model=ImageDescription,
            temperature=0.0,
        )

        result_json = try_extract_json_from_response(response)
        return result_json
    except Exception as e:
        logger.error(f"Error processing with {model_name}: {str(e)}")
        return {"raw_text": f"Error: {str(e)}", "confidence": 0.0}


def process_litellm_based(model_config: ModelConfig, prompt: str, image_url: str) -> Dict[str, Any]:
    """Process an image with a Litellm model."""
    from litellm import completion

    os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": image_url,
                },
            ],
        },
    ]

    try:
        response = completion(
            model=model_config.name,
            messages=messages,
            custom_llm_provider=model_config.provider,
            temperature=0.0,
        )

        result_text = response["choices"][0]["message"]["content"]
        ocr_result = try_extract_json_from_response(result_text)
        return ocr_result
    except Exception as e:
        logger.error(f"Error processing with {model_config.name}: {str(e)}")
        return {"raw_text": f"Error: {str(e)}", "confidence": 0.0}


# ============================================================================
# Core Processing Functions
# ============================================================================


def process_image(
    model_config: ModelConfig, prompt: str, image_base64: str, content_type: str = "image/jpeg"
) -> Dict[str, Any]:
    """Process an image with the specified model."""
    model_name = model_config.name
    image_url = f"data:{content_type};base64,{image_base64}"

    processors = {
        "litellm": lambda: process_litellm_based(model_config, prompt, image_url),
        "ollama": lambda: process_ollama_based(model_name, prompt, image_base64),
        "openai": lambda: process_openai_based(model_name, prompt, image_url),
    }

    processor = processors.get(model_config.ocr_processor)
    if not processor:
        raise ValueError(f"Unsupported ocr_processor: {model_config.ocr_processor}")

    return processor()


def process_single_image(
    model_id: str,
    image: str,
    custom_prompt: Optional[str] = None,
    track_metadata: bool = True,
    index: Optional[int] = None,
) -> Tuple[str, Dict[str, Any], Optional[int]]:
    """Process a single image with a specific model.

    Args:
        model_id: Model ID
        image: Image path or PIL Image
        custom_prompt: Optional custom prompt
        track_metadata: Whether to track metadata with ZenML
        index: Optional index for batch processing

    Returns:
        Tuple of (model_id, result_dict, index)
    """
    start_time = time.time()

    try:
        model_config = MODEL_CONFIGS[model_id]
        content_type, image_base64 = encode_image(image)
        prompt = custom_prompt if custom_prompt else get_prompt()

        result_json = process_image(model_config, prompt, image_base64, content_type)

        processing_time = time.time() - start_time
        if "processing_time" not in result_json:
            result_json["processing_time"] = processing_time

        result_json["model"] = model_id
        result_json["display_name"] = model_config.display
        result_json["ocr_processor"] = model_config.ocr_processor

        return model_id, result_json, index

    except Exception as e:
        processing_time = time.time() - start_time
        error_result = {
            "raw_text": f"Error: {str(e)}",
            "error": str(e),
            "processing_time": processing_time,
            "model": model_id,
            "display_name": MODEL_CONFIGS[model_id].display
            if model_id in MODEL_CONFIGS
            else model_id,
            "ocr_processor": MODEL_CONFIGS[model_id].ocr_processor
            if model_id in MODEL_CONFIGS
            else "unknown",
        }

        return model_id, error_result, index


def process_single_model_task(args):
    """Wrapper function for ThreadPoolExecutor to unpack arguments."""
    return process_single_image(*args)


def process_result_and_track_metrics(
    model_config: ModelConfig,
    result: Dict[str, Any],
    index: int,
    images: List[str],
    results_list: List[Dict[str, Any]],
    processing_times: List[float],
    confidence_scores: List[float],
    track_metadata: bool = True,
):
    """Process a result, log metrics, and track statistics."""
    prefix = model_config.prefix
    display = model_config.display

    image_name = os.path.basename(images[index])
    processing_time = result.get("processing_time", 0)

    formatted_result = {
        "id": index,
        "image_name": image_name,
        "raw_text": result.get("raw_text", "No text found"),
        "processing_time": processing_time,
        "confidence": result.get("confidence", model_config.default_confidence),
    }

    if "error" in result:
        formatted_result["error"] = result["error"]

        if track_metadata:
            log_error_metadata(
                prefix=prefix,
                index=index,
                image_name=image_name,
                error=result["error"],
            )
    else:
        confidence = formatted_result["confidence"]
        if confidence is None:
            confidence = model_config.default_confidence
            formatted_result["confidence"] = confidence

        confidence_scores.append(confidence)

        text_length = len(formatted_result["raw_text"])

        if track_metadata:
            log_image_metadata(
                prefix=prefix,
                index=index,
                image_name=image_name,
                processing_time=processing_time,
                text_length=text_length,
                confidence=confidence,
            )

        logger.info(
            f"{display} OCR [{index + 1}/{len(images)}]: {image_name} - "
            f"{text_length} chars, "
            f"confidence: {confidence:.2f}, "
            f"{processing_time:.2f} seconds"
        )

    results_list.append(formatted_result)
    processing_times.append(processing_time)


# ============================================================================
# Public API Functions
# ============================================================================


def process_models_parallel(
    image_input: Union[str, List[str]],
    model_ids: List[str],
    custom_prompt: Optional[str] = None,
    max_workers: int = 5,
    track_metadata: bool = True,
) -> Dict[str, Any]:
    """Process image(s) with multiple models in parallel.

    Args:
        image_input: Either a single image (path/PIL) or a list of image paths
        model_ids: List of model IDs to process
        custom_prompt: Optional custom prompt
        max_workers: Maximum number of parallel workers
        track_metadata: Whether to track metadata with ZenML

    Returns:
        Dictionary mapping model IDs to their results
    """
    effective_workers = min(len(model_ids), max_workers)
    is_single_image = not isinstance(image_input, list)
    results = {}

    tasks = []
    if is_single_image:
        # For a single image, create tasks for each model
        for model_id in model_ids:
            tasks.append((model_id, image_input, custom_prompt, track_metadata, None))
    else:
        # For multiple images, create tasks for each image/model combination
        for index, image in enumerate(image_input):
            for model_id in model_ids:
                tasks.append((model_id, image, custom_prompt, track_metadata, index))

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = list(executor.map(process_single_model_task, tasks))

        # Process results
        for model_id, result, index in futures:
            if is_single_image:
                # For single image, just store the result
                results[model_id] = result
            else:
                # For multiple images, group results by model
                if model_id not in results:
                    results[model_id] = []
                results[model_id].append((index, result))

    # For multiple images with multiple models, sort results by index
    if not is_single_image:
        for model_id in results:
            results[model_id] = [r for _, r in sorted(results[model_id], key=lambda x: x[0])]

    return results


def process_images_with_model(
    model_config: ModelConfig,
    images: List[str],
    custom_prompt: Optional[str] = None,
    batch_size: int = 5,
    track_metadata: bool = True,
) -> pl.DataFrame:
    """Process multiple images with a specific model configuration.

    Args:
        model_config: Model configuration
        images: List of image paths
        custom_prompt: Optional custom prompt
        batch_size: Number of images to process in parallel
        track_metadata: Whether to track metadata with ZenML

    Returns:
        DataFrame with OCR results
    """
    model_name = model_config.name
    prefix = model_config.prefix
    display = model_config.display

    logger.info(f"Running {display} OCR with model: {model_name}")
    logger.info(f"Processing {len(images)} images with batch size: {batch_size}")

    # Track processing metrics
    results_list = []
    processing_times = []
    confidence_scores = []

    # Process images in batches to control memory usage
    effective_batch_size = min(batch_size, len(images))

    with tqdm(total=len(images), desc=f"Processing with {display}") as pbar:
        for batch_start in range(0, len(images), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(images))
            batch = images[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // effective_batch_size + 1}/"
                f"{(len(images) + effective_batch_size - 1) // effective_batch_size} "
                f"with {len(batch)} images"
            )

            batch_results = process_models_parallel(
                image_input=batch,
                model_ids=[model_name],
                custom_prompt=custom_prompt,
                max_workers=min(effective_batch_size, 10),
                track_metadata=track_metadata,
            )

            if model_name in batch_results:
                for i, result in enumerate(batch_results[model_name]):
                    actual_index = batch_start + i
                    process_result_and_track_metrics(
                        model_config=model_config,
                        result=result,
                        index=actual_index,
                        images=images,
                        results_list=results_list,
                        processing_times=processing_times,
                        confidence_scores=confidence_scores,
                        track_metadata=track_metadata,
                    )
                    pbar.update(1)

    if track_metadata:
        log_summary_metadata(
            prefix=prefix,
            model_name=model_name,
            images_count=len(images),
            processing_times=processing_times,
            confidence_scores=confidence_scores,
        )

    results_df = pl.DataFrame(results_list)
    return results_df


def run_ocr(
    image_input: Union[str, List[str]],
    model_ids: Union[str, List[str]],
    custom_prompt: Optional[str] = None,
    batch_size: int = 5,
    track_metadata: bool = False,
) -> Union[Dict[str, Any], pl.DataFrame, Dict[str, pl.DataFrame]]:
    """Unified interface for running OCR on images with different modes.

    This function handles different combinations of inputs:
    - Single image + single model
    - Single image + multiple models
    - Multiple images + single model
    - Multiple images + multiple models

    Args:
        image_input: Single image path/object or list of image paths
        model_ids: Single model ID or list of model IDs
        custom_prompt: Optional custom prompt
        batch_size: Batch size for parallel processing
        track_metadata: Whether to track metadata with ZenML

    Returns:
        - Single image + single model: Dict result
        - Single image + multiple models: Dict mapping model IDs to results
        - Multiple images + single model: DataFrame with results
        - Multiple images + multiple models: Dict mapping model IDs to DataFrames
    """
    is_single_image = not isinstance(image_input, list)
    is_single_model = not isinstance(model_ids, list)

    if is_single_model:
        model_ids = [model_ids]

    if is_single_image and is_single_model:
        # Single image + single model
        _, result, _ = process_single_image(
            model_id=model_ids[0],
            image=image_input,
            custom_prompt=custom_prompt,
            track_metadata=track_metadata,
        )
        return result

    elif is_single_image and not is_single_model:
        # Single image + multiple models
        return process_models_parallel(
            image_input=image_input,
            model_ids=model_ids,
            custom_prompt=custom_prompt,
            max_workers=min(len(model_ids), 10),
            track_metadata=track_metadata,
        )

    elif not is_single_image and is_single_model:
        # Multiple images + single model
        model_config = MODEL_CONFIGS[model_ids[0]]
        return process_images_with_model(
            model_config=model_config,
            images=image_input,
            custom_prompt=custom_prompt,
            batch_size=batch_size,
            track_metadata=track_metadata,
        )

    else:
        # Multiple images + multiple models
        results = {}
        for model_id in model_ids:
            model_config = MODEL_CONFIGS[model_id]
            results[model_id] = process_images_with_model(
                model_config=model_config,
                images=image_input,
                custom_prompt=custom_prompt,
                batch_size=batch_size,
                track_metadata=track_metadata,
            )
        return results
