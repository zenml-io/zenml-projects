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


def validate_config(config: Dict[str, Any]) -> None:
    """Validate ZenML configuration."""
    # Validate required sections
    if "parameters" not in config:
        raise ValueError("Missing required 'parameters' section in configuration")

    # Validate input parameters
    params = config.get("parameters", {})
    if not params.get("input_image_paths") and not params.get("input_image_folder"):
        raise ValueError(
            "Either parameters.input_image_paths or parameters.input_image_folder must be provided"
        )

    # Validate input folder exists
    image_folder = params.get("input_image_folder")
    if image_folder and not os.path.isdir(image_folder):
        raise ValueError(f"Image folder does not exist: {image_folder}")

    # Validate steps configuration if present
    steps = config.get("steps", {})

    # Validate model configuration
    if "ocr_processor" in steps:
        if "parameters" not in steps["ocr_processor"]:
            raise ValueError("Missing parameters section in steps.ocr_processor")

        if "selected_models" not in params:
            raise ValueError("Missing selected_models in parameters")

    # Validate result saving configuration
    if "result_saver" in steps:
        if "parameters" not in steps["result_saver"]:
            raise ValueError("Missing parameters section in steps.result_saver")


def override_config_with_cli_args(
    config: Dict[str, Any], cli_args: Dict[str, Any]
) -> Dict[str, Any]:
    """Override configuration with command-line arguments."""
    # Deep copy the config to avoid modifying the original
    config = {**config}

    # Ensure parameters section exists
    if "parameters" not in config:
        config["parameters"] = {}

    # Ensure steps section exists
    if "steps" not in config:
        config["steps"] = {}

    # Override input configuration
    if cli_args.get("image_paths"):
        config["parameters"]["input_image_paths"] = cli_args["image_paths"]
    if cli_args.get("image_folder"):
        config["parameters"]["input_image_folder"] = cli_args["image_folder"]

    # Override model configuration
    if cli_args.get("custom_prompt") and "ocr_processor" not in config["steps"]:
        config["steps"]["ocr_processor"] = {"parameters": {}}

    if cli_args.get("custom_prompt"):
        config["steps"]["ocr_processor"]["parameters"]["custom_prompt"] = cli_args["custom_prompt"]

    # Override output configuration
    if cli_args.get("save_results"):
        if "result_saver" not in config["steps"]:
            config["steps"]["result_saver"] = {"parameters": {}}
        config["steps"]["result_saver"]["parameters"]["save_results"] = cli_args["save_results"]

    if cli_args.get("results_directory"):
        if "result_saver" not in config["steps"]:
            config["steps"]["result_saver"] = {"parameters": {}}
        config["steps"]["result_saver"]["parameters"]["results_directory"] = cli_args[
            "results_directory"
        ]

    if cli_args.get("save_visualizations"):
        if "visualizer" not in config["steps"]:
            config["steps"]["visualizer"] = {"parameters": {}}
        config["steps"]["visualizer"]["parameters"]["save_visualizations"] = cli_args[
            "save_visualizations"
        ]

    if cli_args.get("visualization_directory"):
        if "visualizer" not in config["steps"]:
            config["steps"]["visualizer"] = {"parameters": {}}
        config["steps"]["visualizer"]["parameters"]["visualization_directory"] = cli_args[
            "visualization_directory"
        ]

    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the ZenML configuration."""
    print("\n===== OCR Pipeline Configuration =====")

    # Get parameters
    params = config.get("parameters", {})

    # Print pipeline mode
    mode = params.get("mode", "evaluation")
    print(f"Pipeline mode: {mode}")

    # Print model information
    selected_models = params.get("selected_models", [])
    if selected_models:
        print(f"Selected models: {', '.join(selected_models)}")

    # Print input information
    image_paths = params.get("input_image_paths", [])
    if image_paths:
        print(f"Input images: {len(image_paths)} specified")

    image_folder = params.get("input_image_folder")
    if image_folder:
        print(f"Input folder: {image_folder}")

    # Print ground truth information if available
    steps = config.get("steps", {})
    evaluator = steps.get("result_evaluator", {}).get("parameters", {})
    gt_folder = evaluator.get("ground_truth_folder")
    if gt_folder:
        print(f"Ground truth folder: {gt_folder}")
        gt_files = list_available_ground_truth_files(directory=gt_folder)
        print(f"Found {len(gt_files)} ground truth text files")

    # Print output information
    result_saver = steps.get("result_saver", {}).get("parameters", {})
    if result_saver.get("save_results", False):
        print(f"Results will be saved to: {result_saver.get('results_directory', 'ocr_results')}")

    visualizer = steps.get("visualizer", {}).get("parameters", {})
    if visualizer.get("save_visualizations", False):
        print(
            f"Visualizations will be saved to: {visualizer.get('visualization_directory', 'visualizations')}"
        )

    print("=" * 40 + "\n")


def get_image_paths(directory: str) -> List[str]:
    """Get all image paths from a directory."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, ext)))

    return sorted(image_paths)


def list_available_ground_truth_files(directory: Optional[str] = None) -> List[str]:
    """List available ground truth text files in the given directory."""
    if not directory or not os.path.isdir(directory):
        return []

    text_files = glob.glob(os.path.join(directory, "*.txt"))
    return sorted(text_files)
