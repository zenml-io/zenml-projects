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
"""Utilities for handling configuration."""

import glob
import os
from typing import Any, Dict, List, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration file: {e}")


def validate_batch_config(config: Dict[str, Any]) -> None:
    """Validate batch pipeline configuration."""
    if "steps" not in config:
        raise ValueError("Missing required 'steps' section in batch configuration")

    steps = config.get("steps", {})

    if "load_images" not in steps:
        raise ValueError("Missing required 'load_images' step in batch pipeline configuration")

    if "run_ocr" not in steps:
        raise ValueError("Missing required 'run_ocr' step in batch pipeline configuration")

    load_images_params = steps.get("load_images", {}).get("parameters", {})
    if not load_images_params.get("image_folder") and not load_images_params.get("image_paths"):
        raise ValueError(
            "Either image_folder or image_paths must be provided in load_images.parameters"
        )

    image_folder = load_images_params.get("image_folder")
    if image_folder and not os.path.isdir(image_folder):
        raise ValueError(f"Image folder does not exist: {image_folder}")

    run_ocr_params = steps.get("run_ocr", {}).get("parameters", {})
    if "models" not in run_ocr_params or not run_ocr_params["models"]:
        raise ValueError("At least one model must be specified in run_ocr.parameters.models")

    if "models_registry" not in config or not config["models_registry"]:
        raise ValueError("models_registry section is required with at least one model definition")


def validate_evaluation_config(config: Dict[str, Any]) -> None:
    """Validate evaluation pipeline configuration."""
    if "steps" not in config:
        raise ValueError("Missing required 'steps' section in evaluation configuration")

    steps = config.get("steps", {})

    if "load_ocr_results" not in steps:
        raise ValueError(
            "Missing required 'load_ocr_results' step in evaluation pipeline configuration"
        )

    if "load_ground_truth_texts" not in steps:
        raise ValueError(
            "Missing required 'load_ground_truth_texts' step in evaluation pipeline configuration"
        )

    gt_params = steps.get("load_ground_truth_texts", {}).get("parameters", {})
    gt_folder = gt_params.get("ground_truth_folder")
    if gt_folder and not os.path.isdir(gt_folder):
        raise ValueError(f"Ground truth folder does not exist: {gt_folder}")


def override_batch_config(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Override batch pipeline configuration with command-line arguments."""
    modified_config = {**config}

    steps = modified_config.get("steps", {})

    if "load_images" not in steps:
        steps["load_images"] = {"parameters": {}}
    elif "parameters" not in steps["load_images"]:
        steps["load_images"]["parameters"] = {}

    if cli_args.get("image_paths"):
        steps["load_images"]["parameters"]["image_paths"] = cli_args["image_paths"]

    if cli_args.get("image_folder"):
        steps["load_images"]["parameters"]["image_folder"] = cli_args["image_folder"]

    if "run_ocr" not in steps:
        steps["run_ocr"] = {"parameters": {}}
    elif "parameters" not in steps["run_ocr"]:
        steps["run_ocr"]["parameters"] = {}

    if cli_args.get("custom_prompt"):
        steps["run_ocr"]["parameters"]["custom_prompt"] = cli_args["custom_prompt"]

    return modified_config


def override_evaluation_config(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Override evaluation pipeline configuration with command-line arguments."""
    modified_config = {**config}

    steps = modified_config.get("steps", {})

    if "load_ground_truth_texts" not in steps:
        steps["load_ground_truth_texts"] = {"parameters": {}}
    elif "parameters" not in steps["load_ground_truth_texts"]:
        steps["load_ground_truth_texts"]["parameters"] = {}

    if cli_args.get("ground_truth_dir"):
        steps["load_ground_truth_texts"]["parameters"]["ground_truth_folder"] = cli_args[
            "ground_truth_dir"
        ]

    return modified_config


def print_batch_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the batch pipeline configuration."""
    steps = config.get("steps", {})

    print("\n===== Batch OCR Pipeline Configuration =====")
    print(f"Build: {config.get('build', 'N/A')}")
    print(f"Run name: {config.get('run_name', 'N/A')}")

    # Print caching and logging info
    print(f"Cache enabled: {config.get('enable_cache', False)}")
    print(f"Step logs enabled: {config.get('enable_step_logs', False)}")

    # Print input information
    load_params = steps.get("load_images", {}).get("parameters", {})

    image_paths = load_params.get("image_paths", [])
    if image_paths:
        print(f"Input images: {len(image_paths)} specified")

    image_folder = load_params.get("image_folder")
    if image_folder:
        print(f"Input folder: {image_folder}")
        try:
            num_images = len(get_image_paths(image_folder))
            print(f"Found {num_images} images in folder")
        except Exception as e:
            print(f"Unable to access image folder: {e}")

    # Print model information
    run_ocr_params = steps.get("run_ocr", {}).get("parameters", {})
    models = run_ocr_params.get("models", [])
    if models:
        print(f"Models to run: {', '.join(models)}")

    # Print custom prompt if provided
    custom_prompt = run_ocr_params.get("custom_prompt")
    if custom_prompt:
        print(f"Custom prompt: {custom_prompt}")

    # List models from registry
    models_registry = config.get("models_registry", [])
    if models_registry:
        print(f"\nModels in registry: {len(models_registry)}")
        for model in models_registry:
            print(f"  - {model.get('name')} (shorthand: {model.get('shorthand')})")

    print("=" * 40 + "\n")


def print_evaluation_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the evaluation pipeline configuration."""
    steps = config.get("steps", {})

    print("\n===== OCR Evaluation Pipeline Configuration =====")
    print(f"Build: {config.get('build', 'N/A')}")
    print(f"Run name: {config.get('run_name', 'N/A')}")

    # Print caching and logging info
    print(f"Cache enabled: {config.get('enable_cache', False)}")
    print(f"Step logs enabled: {config.get('enable_step_logs', False)}")

    # Print OCR results information
    load_results_params = steps.get("load_ocr_results", {}).get("parameters", {})
    artifact_name = load_results_params.get("artifact_name", "ocr_results")
    artifact_version = load_results_params.get("version", "latest")
    print(f"Loading OCR results from: {artifact_name} (version: {artifact_version})")

    # Print ground truth information
    gt_params = steps.get("load_ground_truth_texts", {}).get("parameters", {})
    gt_folder = gt_params.get("ground_truth_folder")
    if gt_folder:
        print(f"Ground truth folder: {gt_folder}")
        gt_files = list_available_ground_truth_files(directory=gt_folder)
        print(f"Found {len(gt_files)} ground truth text files")

    gt_files = gt_params.get("ground_truth_files", [])
    if gt_files:
        print(f"Using {len(gt_files)} specific ground truth files")

    print("=" * 45 + "\n")


def print_config_summary(
    config: Dict[str, Any],
    is_evaluation_config: bool = False,
) -> None:
    """Print a summary of the ZenML configuration."""
    if is_evaluation_config:
        print_evaluation_config_summary(config)
    else:
        print_batch_config_summary(config)


def get_image_paths(directory: str) -> List[str]:
    """Get all image paths from a directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))

    return sorted(image_paths)


def list_available_ground_truth_files(directory: Optional[str] = "ground_truth_texts") -> List[str]:
    """List available ground truth text files in the given directory.

    Args:
        directory: Directory containing ground truth text files

    Returns:
        List of paths to available ground truth text files
    """
    if not directory or not os.path.isdir(directory):
        return []

    text_files = glob.glob(os.path.join(directory, "*.txt"))
    return sorted(text_files)


def select_config_path(evaluation_mode: bool) -> str:
    """Select the appropriate configuration file path based on the pipeline mode.

    Args:
        evaluation_mode: Whether to use evaluation pipeline configuration

    Returns:
        Path to the configuration file
    """
    if evaluation_mode:
        return "configs/evaluation_pipeline.yaml"
    else:
        return "configs/batch_pipeline.yaml"
