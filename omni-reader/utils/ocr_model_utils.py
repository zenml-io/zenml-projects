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
from typing import Dict, List, Optional

import ollama
import polars as pl
from dotenv import load_dotenv
from zenml import log_metadata
from zenml.logger import get_logger

from utils.encode_image import encode_image
from utils.model_info import ModelConfig
from utils.prompt import ImageDescription, get_prompt

load_dotenv()
logger = get_logger(__name__)


def try_extract_json_from_response(response: str) -> Dict:
    """Extract JSON from a response string.

    Args:
        response: The response string

    Returns:
        Dict with JSON data
    """
    JSON_PATTERN = re.compile(r"```json\n(.*?)```", re.DOTALL)
    DIRECT_JSON_PATTERN = re.compile(r"\{[^}]*\}", re.DOTALL)

    try:
        if match := JSON_PATTERN.search(response):
            json_results = match.group(1)
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(json_results)
        if match := DIRECT_JSON_PATTERN.search(response):
            json_text = match.group(0)
            with contextlib.suppress(json.JSONDecodeError):
                return json.loads(json_text)
    except json.JSONDecodeError:
        error_msg = f"Failed to parse model response: {response[:100]}..."
        logger.error(error_msg)
        return {"raw_text": f"Error: {error_msg}", "confidence": 0.0, "success": False}


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


def process_with_ollama(
    model_name: str,
    image_path: str,
    prompt: str,
    model_config: ModelConfig,
) -> Dict:
    """Process an image with Ollama.

    Args:
        model_name: Name of the Ollama model
        image_path: Path to the image file
        prompt: Prompt text
        model_config: Model configuration

    Returns:
        Dict with OCR results
    """
    _, image_base64 = encode_image(image_path)

    ollama_params = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
                "format": ImageDescription.model_json_schema(),
            }
        ],
    }

    if model_config.additional_params:
        ollama_params.update(model_config.additional_params)

    try:
        response = ollama.chat(**ollama_params)
        result = try_extract_json_from_response(response.message.content)
        return result
    except Exception as e:
        error_msg = f"Error with Ollama OCR: {str(e)}"
        logger.error(error_msg)
        return {
            "raw_text": f"Error: {error_msg}",
            "confidence": 0.0,
        }


def process_with_client(
    client,
    model_name: str,
    image_path: str,
    prompt: str,
    model_config: ModelConfig,
) -> ImageDescription | Dict:
    """Process images with an API client (OpenAI, Mistral, etc.).

    Args:
        client: API client
        model_name: Name of the model
        image_path: Path to the image file
        prompt: Prompt text
        model_config: Model configuration

    Returns:
        API response processed into ImageDescription or Dict
    """
    content_type, image_base64 = encode_image(image_path)

    params = {
        "model": model_name,
        "response_model": ImageDescription,
        **({"max_tokens": model_config.max_tokens} if model_config.max_tokens else {}),
        **(model_config.additional_params or {}),
    }

    return client.chat.completions.create(
        **params,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{content_type};base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
    )


def process_images_with_model(
    model_config: ModelConfig,
    images: List[str],
    custom_prompt: Optional[str] = None,
) -> pl.DataFrame:
    """Process images with a specific model configuration.

    Args:
        model_config: Model configuration
        images: List of image paths
        custom_prompt: Optional custom prompt

    Returns:
        DataFrame with OCR results
    """
    model_name = model_config.name
    prefix = model_config.prefix
    prompt = custom_prompt if custom_prompt else get_prompt()

    logger.info(f"Running {prefix.capitalize()} OCR with model: {model_name}")
    logger.info(f"Processing {len(images)} images")

    results_list = []
    processing_times = []
    confidence_scores = []

    if "ollama" not in model_name:
        client = model_config.client_factory()

    for i, image_path in enumerate(images):
        start_time = time.time()
        image_name = os.path.basename(image_path)

        try:
            if "gemma" in model_name:
                response = process_with_ollama(model_name, image_path, prompt, model_config)
            else:
                response = process_with_client(client, model_name, image_path, prompt, model_config)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            confidence = response.confidence

            confidence = max(confidence, model_config.default_confidence)
            confidence_scores.append(confidence)

            result = {
                "id": i,
                "image_name": image_name,
                "raw_text": response.raw_text if response.raw_text else "No text found",
                "processing_time": processing_time,
                "confidence": confidence,
            }

            # Log individual image metadata
            log_image_metadata(
                prefix=prefix,
                index=i,
                image_name=image_name,
                processing_time=processing_time,
                text_length=len(result["raw_text"]),
                confidence=confidence,
            )

            logger.info(
                f"{prefix.capitalize()} OCR [{i + 1}/{len(images)}]: {image_name} - "
                f"{len(result['raw_text'])} chars, "
                f"confidence: {confidence:.2f}, "
                f"{processing_time:.2f} seconds"
            )

        except Exception as e:
            error_message = f"An unexpected error occurred on image {image_name}: {str(e)}"
            logger.error(error_message)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            result = {
                "id": i,
                "image_name": image_name,
                "raw_text": "Error: Failed to extract text",
                "processing_time": processing_time,
                "confidence": 0.0,
                "error": error_message,
            }

            # Log error metadata
            log_error_metadata(
                prefix=prefix,
                index=i,
                image_name=image_name,
                error=str(e),
            )

        results_list.append(result)

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
