# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Unified step for running OCR with a one or multiple models."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import polars as pl
from tqdm import tqdm
from typing_extensions import Annotated
from utils.model_configs import MODEL_CONFIGS
from utils.ocr_processing import process_images_with_model
from utils.visualizations import create_ocr_batch_visualization
from zenml import log_metadata, step
from zenml.logger import get_logger
from zenml.types import HTMLString

logger = get_logger(__name__)


@step()
def run_ocr(
    images: List[str],
    models: List[str],
    custom_prompt: Optional[str] = None,
) -> Tuple[
    Annotated[pl.DataFrame, "ocr_results"],
    Annotated[HTMLString, "ocr_batch_visualization"],
]:
    """Extract text from images using multiple models in parallel.

    Args:
        images: List of paths to image files
        models: List of model names to use
        custom_prompt: Optional custom prompt to override the default prompt

    Returns:
        pl.DataFrame: Combined results from all models with OCR results

    Raises:
        ValueError: If any model_name is not supported
    """
    for model in models:
        if model not in MODEL_CONFIGS:
            supported = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {supported}"
            )

    logger.info(
        f"Running OCR with {len(models)} models on {len(images)} images."
    )

    model_dfs = {}
    performance_metrics = {}

    with ThreadPoolExecutor(max_workers=min(len(models), 5)) as executor:
        futures = {
            model: executor.submit(
                process_images_with_model,
                model_config=MODEL_CONFIGS[model],
                images=images,
                custom_prompt=custom_prompt,
                track_metadata=True,
            )
            for model in models
        }
        with tqdm(total=len(models), desc="Processing models") as pbar:
            for model, future in futures.items():
                start = time.time()
                try:
                    results = future.result()
                    results = results.with_columns(
                        pl.lit(model).alias("model_name"),
                        pl.lit(MODEL_CONFIGS[model].display).alias(
                            "model_display_name"
                        ),
                    )
                    model_dfs[model] = results

                    performance_metrics[model] = {
                        "total_time": time.time() - start,
                        "images_processed": len(images),
                    }
                except Exception as e:
                    logger.error(f"Error processing {model}: {e}")
                    error_df = pl.DataFrame(
                        {
                            "id": list(range(len(images))),
                            "image_name": [
                                os.path.basename(img) for img in images
                            ],
                            "raw_text": [f"Error: {e}"] * len(images),
                            "processing_time": [0.0] * len(images),
                            "confidence": [0.0] * len(images),
                            "error": [str(e)] * len(images),
                            "model_name": [model] * len(images),
                            "model_display_name": [
                                MODEL_CONFIGS[model].display
                            ]
                            * len(images),
                        }
                    )
                    model_dfs[model] = error_df
                    performance_metrics[model] = {
                        "error": str(e),
                        "total_time": time.time() - start,
                        "images_processed": 0,
                    }
                finally:
                    pbar.update(1)

    combined_results = pl.concat(list(model_dfs.values()), how="diagonal")

    # Generate HTML visualization
    html_visualization = create_ocr_batch_visualization(combined_results)

    log_metadata(
        metadata={
            "ocr_results_artifact_name": "ocr_results",
            "ocr_results_artifact_type": "polars.DataFrame",
            "ocr_batch_visualization_artifact_name": "ocr_batch_visualization",
            "ocr_batch_visualization_artifact_type": "zenml.types.HTMLString",
        },
    )

    return combined_results, html_visualization
