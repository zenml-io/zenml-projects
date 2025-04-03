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
"""Run OCR pipeline with or without ZenML tracking.

This module provides two modes of operation:
1. UI Mode: Direct OCR with no metadata/artifact tracking (for Streamlit)
2. Pipeline Mode: Full ZenML pipeline with tracking
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from PIL import Image

from pipelines.batch_pipeline import run_ocr_batch_pipeline
from pipelines.evaluation_pipeline import run_ocr_evaluation_pipeline
from utils.config import (
    get_image_paths,
    list_available_ground_truth_files,
    load_config,
    print_config_summary,
)
from utils.model_configs import DEFAULT_MODEL, MODEL_CONFIGS
from utils.ocr_processing import run_ocr

load_dotenv()


def run_ocr_from_ui(
    image: Union[str, Image.Image],
    model: str,
    custom_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text directly using OCR model without ZenML tracking."""
    start_time = time.time()

    # Get model configuration based on model ID
    if model not in MODEL_CONFIGS:
        return {
            "raw_text": f"Error: Model '{model}' not found in MODEL_CONFIGS",
            "error": f"Invalid model: {model}",
            "processing_time": 0,
            "model": model,
        }

    try:
        # Use the unified run_ocr function with track_metadata=False for UI
        result = run_ocr(
            image_input=image,
            model_ids=model,
            custom_prompt=custom_prompt,
            track_metadata=False,
        )

        # Ensure processing_time is properly set
        if "processing_time" not in result:
            result["processing_time"] = time.time() - start_time

        # Ensure model info is set
        result["model"] = model
        result["display_name"] = MODEL_CONFIGS[model].display
        result["ocr_processor"] = MODEL_CONFIGS[model].ocr_processor

        return result
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "raw_text": f"Error: Failed to extract text - {str(e)}",
            "error": str(e),
            "processing_time": processing_time,
            "model": model,
            "display_name": MODEL_CONFIGS[model].display,
            "ocr_processor": MODEL_CONFIGS[model].ocr_processor,
        }


def run_models_in_parallel(
    image_path: Union[str, Image.Image],
    model_ids: List[str],
    custom_prompt: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Process an image with multiple models in parallel."""
    try:
        # Use the unified run_ocr function for parallel processing
        results = run_ocr(
            image_input=image_path,
            model_ids=model_ids,
            custom_prompt=custom_prompt,
            track_metadata=False,
        )

        # Display progress in CLI mode
        print(f"Processed image with {len(model_ids)} models")
        return results
    except Exception as e:
        print(f"Error processing models in parallel: {str(e)}")
        results = {}

        # Create error results for each model
        for model_id in model_ids:
            results[model_id] = {
                "raw_text": f"Error: {str(e)}",
                "error": str(e),
                "processing_time": 0,
                "model": model_id,
            }

        return results


def list_supported_models():
    """List all supported models."""
    print("\nSupported models:")
    print("-" * 70)
    print(f"{'Model ID':<25} {'Display Name':<30} {'OCR Processor':<15}")
    print("-" * 70)

    for model_id, config in MODEL_CONFIGS.items():
        print(f"{model_id:<25} {config.display:<30} {config.ocr_processor:<15}")
        if config.provider:
            print(f"{'Provider':<25} {config.provider:<30}")

    print("\nDefault model:", DEFAULT_MODEL.name)
    print("-" * 70)


def format_model_results(model_id, result):
    """Format results for a specific model."""
    model_config = MODEL_CONFIGS.get(model_id, None)
    model_display = model_config.display if model_config else model_id

    output = f"\n{model_display} results:"

    if "error" in result:
        output += f"\n❌ Error: {result.get('error', 'Unknown error')}"
    else:
        text = result["raw_text"]
        if len(text) > 150:
            text = f"{text[:150]}..."
        output += f"\n✅ Text: {text}"

    output += f"\n⏱️ Processing time: {result.get('processing_time', 0):.2f}s"

    if "confidence" in result and result["confidence"] is not None:
        output += f"\n🎯 Confidence: {result['confidence']:.2%}"

    return output


def run_ui_mode(args, parser):
    """Run the application in streamlit UI mode without ZenML tracking."""
    if args.list_models:
        list_supported_models()
        return

    if args.image_paths:
        image_path = args.image_paths[0]  # Take the first image for UI mode
    elif args.image_folder:
        image_paths = get_image_paths(args.image_folder)
        if not image_paths:
            print(f"No images found in directory: {args.image_folder}")
            return
        image_path = image_paths[0]  # Take the first image for UI mode
    else:
        parser.error("Error: Please provide an image path or folder")
        return

    if not os.path.exists(image_path):
        parser.error(f"Error: Image file '{image_path}' not found.")
        return

    start_time = time.time()

    if args.models == "all":
        # Run all models in parallel
        print(f"Processing image with all {len(MODEL_CONFIGS)} models in parallel...")
        results = run_models_in_parallel(
            image_path,
            list(MODEL_CONFIGS.keys()),
            args.custom_prompt,
        )

        successful_models = sum(1 for result in results.values() if "error" not in result)
        failed_models = len(results) - successful_models

        print("\n" + "=" * 50)
        print(f"OCR COMPARISON RESULTS ({successful_models} successful, {failed_models} failed)")
        print("=" * 50)

        # individual model results
        for model_id, result in results.items():
            print(format_model_results(model_id, result))

        print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
        print("=" * 50)

    elif "," in args.models:
        # Run specific models in parallel
        model_ids = [model_id.strip() for model_id in args.models.split(",")]

        invalid_models = [model_id for model_id in model_ids if model_id not in MODEL_CONFIGS]
        if invalid_models:
            print(f"Error: The following models are not supported: {', '.join(invalid_models)}")
            print("Use --list-models to see all supported models.")
            return

        print(f"Processing image with {len(model_ids)} selected models in parallel...")
        results = run_models_in_parallel(
            image_path,
            model_ids,
            args.custom_prompt,
        )

        successful_models = sum(1 for result in results.values() if "error" not in result)
        failed_models = len(results) - successful_models

        print("\n" + "=" * 50)
        print(f"OCR COMPARISON RESULTS ({successful_models} successful, {failed_models} failed)")
        print("=" * 50)

        # individual model results
        for model_id, result in results.items():
            print(format_model_results(model_id, result))

        print(f"\n⏱️ Total time: {time.time() - start_time:.2f}s")
        print("=" * 50)

    else:
        # Run a single model
        model_id = args.models
        if model_id not in MODEL_CONFIGS:
            print(f"Error: Model '{model_id}' not supported.")
            print("Use --list-models to see all supported models.")
            return

        print(f"\nProcessing with {model_id} model...")
        result = run_ocr_from_ui(
            image_path,
            model_id,
            args.custom_prompt,
        )

        print("\n" + "=" * 50)
        print(f"OCR RESULT FOR {model_id}")
        print("=" * 50)
        print(format_model_results(model_id, result))
        print("=" * 50)


def run_pipeline_mode(args, parser):
    """Run the application in full pipeline mode with ZenML tracking."""
    # List available ground truth files
    if args.list_ground_truth_files:
        gt_files = list_available_ground_truth_files(directory=args.ground_truth_dir)
        if gt_files:
            print("Available ground truth files:")
            for i, file in enumerate(gt_files):
                print(f"  {i + 1}. {file}")
        else:
            print(f"No ground truth files found in '{args.ground_truth_dir}'")
        return

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        parser.error("Please provide a configuration file with --config")
        return

    # Override config with CLI arguments if provided
    if args.image_paths or args.image_folder or args.custom_prompt:
        # Create parameters section if it doesn't exist
        if "parameters" not in config:
            config["parameters"] = {}

        # Update parameters with CLI arguments
        if args.image_paths:
            config["parameters"]["input_image_paths"] = args.image_paths
        if args.image_folder:
            config["parameters"]["input_image_folder"] = args.image_folder
        if args.custom_prompt:
            # Create steps section if needed
            if "steps" not in config:
                config["steps"] = {}
            if "ocr_processor" not in config["steps"]:
                config["steps"]["ocr_processor"] = {"parameters": {}}

            # Set custom prompt
            config["steps"]["ocr_processor"]["parameters"]["custom_prompt"] = args.custom_prompt

    print_config_summary(config)

    # Create output directories if needed
    if (
        "steps" in config
        and "save_ocr_results" in config["steps"]
        and config["steps"]["save_ocr_results"].get("parameters", {}).get("save_results", False)
    ):
        results_dir = config["steps"]["save_ocr_results"]["parameters"].get(
            "results_directory", "ocr_results"
        )
        os.makedirs(results_dir, exist_ok=True)

    if (
        "steps" in config
        and "save_visualization" in config["steps"]
        and config["steps"]["save_visualization"].get("parameters", {}).get("save_locally", False)
    ):
        viz_dir = config["steps"]["save_visualization"]["parameters"].get(
            "visualization_directory", "visualizations"
        )
        os.makedirs(viz_dir, exist_ok=True)

    # Run pipeline in specified mode
    mode = config.get("parameters", {}).get("mode", "evaluation")
    if mode == "batch":
        print("Running OCR Batch Pipeline...")
        run_ocr_batch_pipeline(config)
    else:  # Default to evaluation mode
        print("Running OCR Evaluation Pipeline...")
        run_ocr_evaluation_pipeline(config)


def main():
    """Main entry point for the OCR tool."""
    parser = argparse.ArgumentParser(
        description="Run OCR between vision models with or without ZenML tracking"
    )

    # Mode selection
    parser.add_argument(
        "--ui_mode",
        action="store_true",
        help="Run in UI mode without ZenML tracking (for Streamlit)",
    )

    # Config file options (pipeline mode)
    config_group = parser.add_argument_group("Pipeline Mode Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (for pipeline mode)",
    )
    config_group.add_argument(
        "--create-default-config",
        type=str,
        metavar="PATH",
        help="Create a default configuration file at the specified path and exit",
    )

    # Ground truth utilities (pipeline mode)
    gt_group = parser.add_argument_group("Ground truth utilities (pipeline mode)")
    gt_group.add_argument(
        "--list-ground-truth-files",
        action="store_true",
        help="List available ground truth files and exit",
    )
    gt_group.add_argument(
        "--ground-truth-dir",
        type=str,
        default="ground_truth_texts",
        help="Directory to look for ground truth files (for --list-ground-truth-files)",
    )

    # Quick access options (shared between modes)
    input_group = parser.add_argument_group("Input options (shared)")
    input_group.add_argument(
        "--image-paths",
        nargs="+",
        help="Paths to images to process",
    )
    input_group.add_argument(
        "--image-folder",
        type=str,
        help="Folder containing images to process",
    )
    input_group.add_argument(
        "--custom-prompt",
        type=str,
        dest="custom_prompt",
        help="Custom prompt to use for OCR models",
    )

    # UI mode specific options
    ui_group = parser.add_argument_group("UI Mode Options")
    ui_group.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODEL.name,
        help="Model(s) to use: a specific model ID, 'all' to compare all, or a comma-separated list (for UI mode)",
    )
    ui_group.add_argument(
        "--list-models",
        action="store_true",
        help="List all supported models and exit (for UI mode)",
    )

    args = parser.parse_args()

    if args.ui_mode:
        run_ui_mode(args, parser)
    else:
        run_pipeline_mode(args, parser)


if __name__ == "__main__":
    main()
