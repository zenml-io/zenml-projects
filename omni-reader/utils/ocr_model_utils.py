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
from typing import Dict, List, Optional

import polars as pl
from dotenv import load_dotenv
from zenml import log_metadata
from zenml.logger import get_logger

from utils.encode_image import encode_image
from utils.model_info import ModelConfig
from utils.prompt import ImageDescription, get_prompt

load_dotenv()
logger = get_logger(__name__)


def create_message_with_image(prompt: str, image_path: str, model_prefix: str) -> List[Dict]:
    """Create a message with an image for the model.

    Args:
        prompt: The text prompt
        image_path: Path to the image file
        model_prefix: The model prefix (openai, mistral, etc.)

    Returns:
        List of message dictionaries ready for the model
    """
    content_type, image_base64 = encode_image(image_path)

    return [
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
    ]


def log_image_metadata(
    prefix: str,
    index: int,
    image_name: str,
    processing_time: float,
    text_length: int,
    confidence: float,
    running_from_ui: bool = False,
):
    """Log metadata for a processed image.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        index: Image index
        image_name: Name of the image file
        processing_time: Processing time in seconds
        text_length: Length of extracted text
        confidence: Confidence score
        running_from_ui: Whether running from UI (affects logging)
    """
    if running_from_ui:
        return

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
    running_from_ui: bool = False,
):
    """Log error metadata for a failed image processing.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        index: Image index
        image_name: Name of the image file
        error: Error message
        running_from_ui: Whether running from UI (affects logging)
    """
    if running_from_ui:
        return

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
    running_from_ui: bool = False,
):
    """Log summary metadata for all processed images.

    Args:
        prefix: The model prefix (openai, mistral, etc.)
        model_name: Name of the model
        images_count: Number of images processed
        processing_times: List of processing times
        confidence_scores: List of confidence scores
        running_from_ui: Whether running from UI (affects logging)
    """
    if running_from_ui or not processing_times:
        return

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
    running_from_ui: bool = False,
) -> pl.DataFrame:
    """Process images with a specific model configuration.

    Args:
        model_config: Model configuration
        images: List of image paths
        custom_prompt: Optional custom prompt
        running_from_ui: Whether running from UI

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

    client = model_config.client_factory()

    for i, image_path in enumerate(images):
        start_time = time.time()
        image_name = os.path.basename(image_path)

        try:
            messages = create_message_with_image(prompt, image_path, prefix)

            # Prepare params based on model config
            params = {
                "model": model_name,
                "response_model": ImageDescription,
                "messages": messages,
            }

            # Add model-specific parameters if provided
            if model_config.max_tokens:
                params["max_tokens"] = model_config.max_tokens

            if model_config.additional_params:
                params.update(model_config.additional_params)

            response = client.chat.completions.create(**params)

            processing_time = time.time() - start_time
            processing_times.append(processing_time)

            confidence = response.confidence

            # Apply default minimum confidence if needed
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
                running_from_ui=running_from_ui,
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
                running_from_ui=running_from_ui,
            )

        results_list.append(result)

    # Log summary statistics
    log_summary_metadata(
        prefix=prefix,
        model_name=model_name,
        images_count=len(images),
        processing_times=processing_times,
        confidence_scores=confidence_scores,
        running_from_ui=running_from_ui,
    )

    # Convert to polars dataframe
    results_df = pl.DataFrame(results_list)
    return results_df
