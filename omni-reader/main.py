"""Main script to run the OCR comparison pipeline."""

import argparse
import glob
import os
from typing import List

from pipelines.ocr_pipeline import ocr_comparison_pipeline


def get_image_paths(directory: str) -> List[str]:
    """Get all image paths from a directory.

    Args:
        directory: Directory containing images

    Returns:
        List of image paths
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))

    return sorted(image_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR comparison pipeline")
    parser.add_argument(
        "--image_dir",
        type=str,
        default="assets",
        help="Directory containing images to process (default: assets)",
    )
    parser.add_argument("--custom_prompt", type=str, help="Custom prompt to use for OCR models")
    parser.add_argument(
        "--ground_truth",
        type=str,
        nargs="+",
        help="Ground truth texts for evaluation (provide one for each image in the same order)",
    )
    parser.add_argument("--config_path", type=str, help="Path to YAML config file")

    args = parser.parse_args()

    # Get images from directory
    image_paths = get_image_paths(args.image_dir)

    if not image_paths:
        print(f"No images found in directory: {args.image_dir}")
        exit(1)

    print(f"Found {len(image_paths)} images in {args.image_dir}")

    # Configure pipeline if config file provided
    if args.config_path:
        ocr_comparison_pipeline = ocr_comparison_pipeline.with_options(config_path=args.config_path)

    # Run pipeline
    pipeline_run = ocr_comparison_pipeline(
        images=image_paths,
        custom_prompt=args.custom_prompt,
    )

    print(f"\nPipeline run completed: {pipeline_run.id}")
