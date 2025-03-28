"""OCR Pipeline implementation."""

import os
from typing import Annotated, Dict, List, Optional, Tuple

import polars as pl
from PIL import Image
from zenml import log_metadata, pipeline

from steps.evaluate_models import evaluate_models
from steps.run_gemma3_ocr import run_gemma3_ocr
from steps.run_mistral_ocr import run_mistral_ocr
from utils import DOCKER_SETTINGS, MLFLOW_SETTINGS


@pipeline(
    settings={
        "docker": DOCKER_SETTINGS,
    },
    name="ocr_comparison_pipeline",
    enable_cache=False,
)
def ocr_comparison_pipeline(
    images: List[str],
    custom_prompt: Optional[str] = None,
    ground_truth_texts: Optional[List[str]] = None,
):
    """Run OCR comparison pipeline between Gemma3 and Mistral models.

    Args:
        images: List of paths to images to process
        custom_prompt: Optional custom prompt to use for both models
        ground_truth_texts: Optional list of ground truth texts for evaluation
    """
    # Initialize dataframes to store results
    gemma_results_list = []
    mistral_results_list = []
    ground_truth_list = []

    for i, image_path in enumerate(images):
        image = Image.open(image_path)

        # Get image name from path for logging
        image_name = os.path.basename(image_path)

        # Run OCR with both models
        gemma_result = run_gemma3_ocr(image=image_path, custom_prompt=custom_prompt)
        mistral_result = run_mistral_ocr(image=image_path, custom_prompt=custom_prompt)

        # Create entries for dataframes
        gemma_entry = {
            "id": i,
            "image_name": image_name,
            "gemma_text": gemma_result["raw_text"],
            "gemma_entities": ", ".join(gemma_result.get("entities", [])),
            "gemma_processing_time": gemma_result.get("processing_time", 0),
        }

        mistral_entry = {
            "id": i,
            "image_name": image_name,
            "mistral_text": mistral_result["raw_text"],
            "mistral_entities": ", ".join(mistral_result.get("entities", [])),
            "mistral_processing_time": mistral_result.get("processing_time", 0),
        }

        # Log metadata about this image processing
        log_metadata(
            metadata={
                f"image_{i}": {
                    "name": image_name,
                    "gemma_entities_count": len(gemma_result.get("entities", [])),
                    "mistral_entities_count": len(mistral_result.get("entities", [])),
                }
            }
        )

        gemma_results_list.append(gemma_entry)
        mistral_results_list.append(mistral_entry)

        # Add ground truth if available
        if ground_truth_texts and i < len(ground_truth_texts):
            ground_truth_list.append(
                {
                    "id": i,
                    "image_name": image_name,
                    "ground_truth_text": ground_truth_texts[i],
                }
            )

    # Convert to polars dataframes
    gemma_df = pl.DataFrame(gemma_results_list)
    mistral_df = pl.DataFrame(mistral_results_list)

    # Create ground truth dataframe if available
    ground_truth_df = None
    if ground_truth_list:
        ground_truth_df = pl.DataFrame(ground_truth_list)

    # Evaluate models
    evaluation_results = evaluate_models(
        gemma_results=gemma_df,
        mistral_results=mistral_df,
        ground_truth=ground_truth_df,
    )

    return evaluation_results
