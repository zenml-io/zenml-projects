"""Steps for saving OCR data."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import polars as pl
from zenml import log_metadata, step
from zenml.logger import get_logger
from zenml.types import HTMLString

from utils.io_utils import save_ground_truth_to_json, save_ocr_data_to_json

logger = get_logger(__name__)


@step
def save_ground_truth(
    ground_truth: Dict[str, pl.DataFrame],
    output_dir: str = "ground_truth",
    suffix: Optional[str] = None,
) -> str:
    """Save ground truth data to a JSON file.

    Args:
        ground_truth: Dictionary containing ground truth results
        output_dir: Directory to save ground truth data
        suffix: Optional suffix to add to the filename

    Returns:
        Path to the saved ground truth file
    """
    prefix = "gt"
    if suffix:
        prefix = f"{prefix}_{suffix}"

    filepath = save_ground_truth_to_json(ground_truth, output_dir, prefix)

    logger.info(f"Ground truth saved to: {filepath}")

    log_metadata(
        metadata={
            "ground_truth_saved": {
                "path": filepath,
                "count": len(ground_truth["ground_truth_results"]) if "ground_truth_results" in ground_truth else 0,
            }
        }
    )

    return filepath


@step
def save_ocr_results(
    gemma_results: Dict[str, pl.DataFrame],
    mistral_results: Dict[str, pl.DataFrame],
    output_dir: str = "ocr_results",
) -> Tuple[str, str]:
    """Save OCR results from both models.

    Args:
        gemma_results: Dictionary containing Gemma OCR results
        mistral_results: Dictionary containing Mistral OCR results
        output_dir: Base directory to save OCR results

    Returns:
        Tuple of paths to saved Gemma and Mistral results
    """
    # Create model-specific output directories
    gemma_dir = os.path.join(output_dir, "gemma")
    mistral_dir = os.path.join(output_dir, "mistral")

    # Save Gemma results
    gemma_path = save_ocr_data_to_json(
        data=gemma_results, output_dir=gemma_dir, prefix="gemma", key_name="gemma_results"
    )

    # Save Mistral results
    mistral_path = save_ocr_data_to_json(
        data=mistral_results, output_dir=mistral_dir, prefix="mistral", key_name="mistral_results"
    )

    # Log metadata
    log_metadata(
        metadata={
            "ocr_results_saved": {
                "gemma_path": gemma_path,
                "mistral_path": mistral_path,
                "gemma_count": len(gemma_results["gemma_results"]),
                "mistral_count": len(mistral_results["mistral_results"]),
            }
        }
    )

    logger.info(f"Gemma results saved to: {gemma_path}")
    logger.info(f"Mistral results saved to: {mistral_path}")

    return gemma_path, mistral_path


@step
def save_visualization(
    visualization: HTMLString,
    output_dir: str = "visualizations",
) -> str:
    """Save HTML visualization to a file.

    Args:
        visualization: HTML visualization content
        output_dir: Directory to save visualization

    Returns:
        Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_visualization_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    # Save HTML to file
    with open(filepath, "w") as f:
        f.write(str(visualization))

    logger.info(f"Visualization saved to: {filepath}")

    # Log metadata
    log_metadata(
        metadata={
            "visualization_saved": {
                "path": filepath,
            }
        }
    )

    return filepath
