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

import contextlib
import json
import os
import re
import statistics
import time
from typing import Any, Dict, List, Optional

import polars as pl
from dotenv import load_dotenv
from zenml import log_metadata
from zenml.logger import get_logger

from utils.encode_image import encode_image
from utils.model_configs import ModelConfig
from utils.prompt import ImageDescription, get_prompt

load_dotenv()
logger = get_logger(__name__)


def try_extract_json_from_response(response: Any) -> Dict:
    """Extract JSON from a response, handling various formats.

    Args:
        response: The response which could be string, dict, or object

    Returns:
        Dict with extracted data
    """
    # If already a dict with raw_text, return it
    if isinstance(response, dict) and "raw_text" in response:
        return response

    # Convert to string if it's an object with content
    response_text = ""
    if hasattr(response, "choices") and len(response.choices) > 0:
        if hasattr(response.choices[0], "message") and hasattr(
            response.choices[0].message, "content"
        ):
            response_text = response.choices[0].message.content
    elif isinstance(response, str):
        response_text = response
    elif hasattr(response, "raw_text"):
        # This handles the ImageDescription object case
        return {"raw_text": response.raw_text, "confidence": getattr(response, "confidence", None)}

    # Try to extract JSON from the text
    JSON_PATTERN = re.compile(r"```json\n(.*?)```", re.DOTALL)
    DIRECT_JSON_PATTERN = re.compile(r"\{[^}]*\}", re.DOTALL)

    try:
        if match := JSON_PATTERN.search(response_text):
            json_results = match.group(1)
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(json_results)
        if match := DIRECT_JSON_PATTERN.search(response_text):
            json_text = match.group(0)
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(json_text)

        # If we get here, no JSON could be extracted, so use the text as raw_text
        return {"raw_text": response_text, "confidence": None}
    except Exception as e:
        # Fallback for any other errors
        return {"raw_text": f"Error: {str(e)}", "confidence": 0.0, "success": False}


def log_image_metadata(
    prefix: str,
    index: int,
    image_name: str,
    processing_time: float,
    text_length: int,
    confidence: float,
):
    """Log metadata for a processed image.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        index: Image index
        image_name: Name of the image file
        processing_time: Processing time in seconds
        text_length: Length of extracted text
        confidence: Confidence score
    """
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
    """Log error metadata for a failed image processing.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        index: Image index
        image_name: Name of the image file
        error: Error message
    """
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
    """Log summary metadata for all processed images.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        model_name: Name of the model
        images_count: Number of images processed
        processing_times: List of processing times
        confidence_scores: List of confidence scores
    """
    avg_time = statistics.mean(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)

    avg_confidence = 0.0
    if confidence_scores:
        avg_confidence = statistics.mean(confidence_scores)

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


def process_images_with_model(
    model_config: ModelConfig,
    images: List[str],
    custom_prompt: Optional[str] = None,
    batch_size: int = 5,
) -> pl.DataFrame:
    """Process images with a specific model configuration.

    Args:
        model_config: Model configuration
        images: List of image paths
        custom_prompt: Optional custom prompt
        batch_size: Number of images to process in parallel (default: 5)

    Returns:
        DataFrame with OCR results
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    model_name = model_config.name
    prefix = model_config.prefix
    display = model_config.display
    prompt = custom_prompt if custom_prompt else get_prompt()

    logger.info(f"Running {display} OCR with model: {model_name}")
    logger.info(f"Processing {len(images)} images with batch size: {batch_size}")

    results_list = []
    processing_times = []
    confidence_scores = []

    def process_single_image(args):
        i, image_path = args
        start_time = time.time()
        image_name = os.path.basename(image_path)

        try:
            content_type, image_base64 = encode_image(image_path)

            result_json = model_config.process_image(prompt, image_base64, content_type)

            raw_text = result_json.get("raw_text", "No text found")
            confidence = result_json.get("confidence", model_config.default_confidence)
            if confidence is None:
                confidence = model_config.default_confidence

            processing_time = time.time() - start_time

            result = {
                "id": i,
                "image_name": image_name,
                "raw_text": raw_text,
                "processing_time": processing_time,
                "confidence": confidence,
            }

            log_image_metadata(
                prefix=prefix,
                index=i,
                image_name=image_name,
                processing_time=processing_time,
                text_length=len(result["raw_text"]),
                confidence=confidence,
            )

            logger.info(
                f"{display} OCR [{i + 1}/{len(images)}]: {image_name} - "
                f"{len(result['raw_text'])} chars, "
                f"confidence: {confidence:.2f}, "
                f"{processing_time:.2f} seconds"
            )

            return {
                "result": result,
                "processing_time": processing_time,
                "confidence": confidence,
                "success": True,
            }

        except Exception as e:
            error_message = f"An unexpected error occurred on image {image_name}: {str(e)}"
            logger.error(error_message)
            processing_time = time.time() - start_time

            log_error_metadata(
                prefix=prefix,
                index=i,
                image_name=image_name,
                error=str(e),
            )

            return {
                "result": {
                    "id": i,
                    "image_name": image_name,
                    "raw_text": f"Error: Failed to extract text - {str(e)}",
                    "processing_time": processing_time,
                    "confidence": 0.0,
                    "error": error_message,
                },
                "processing_time": processing_time,
                "confidence": 0.0,
                "success": False,
            }

    effective_batch_size = min(batch_size, len(images))
    max_workers = min(effective_batch_size, 10)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(images), desc=f"Processing with {display}") as pbar:
            image_batches = [
                images[i : i + effective_batch_size]
                for i in range(0, len(images), effective_batch_size)
            ]

            for batch_index, batch in enumerate(image_batches):
                logger.info(
                    f"Processing batch {batch_index + 1}/{len(image_batches)} with {len(batch)} images"
                )

                batch_indices = range(
                    batch_index * effective_batch_size,
                    batch_index * effective_batch_size + len(batch),
                )

                batch_futures = list(executor.map(process_single_image, zip(batch_indices, batch)))

                for result_dict in batch_futures:
                    results_list.append(result_dict["result"])
                    processing_times.append(result_dict["processing_time"])

                    if result_dict["success"]:
                        confidence_scores.append(result_dict["confidence"])

                    pbar.update(1)

    # Log summary statistics
    log_summary_metadata(
        prefix=prefix,
        model_name=model_name,
        images_count=len(images),
        processing_times=processing_times,
        confidence_scores=confidence_scores,
    )

    # Convert to polars dataframe
    results_df = pl.DataFrame(results_list)
    return results_df
