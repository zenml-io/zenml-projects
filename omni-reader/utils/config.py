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

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file is not valid YAML
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration file: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration.

    Args:
        config: Dictionary containing configuration

    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate top-level sections
    required_sections = ["input", "models", "ground_truth", "output"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in configuration")

    # Validate input section
    if not config["input"].get("image_paths") and not config["input"].get("image_folder"):
        raise ValueError("Either input.image_paths or input.image_folder must be provided")

    if config["input"].get("image_folder") and not os.path.isdir(config["input"].get("image_folder")):
        raise ValueError(f"Image folder does not exist: {config['input'].get('image_folder')}")

    # Validate ground truth configuration
    gt_source = config["ground_truth"].get("source", "none")
    if gt_source not in ["openai", "manual", "file", "none"]:
        raise ValueError(f"Invalid ground_truth.source: {gt_source}. Must be one of: openai, manual, file, none")

    if gt_source == "manual" and not config["ground_truth"].get("texts"):
        raise ValueError("When using ground_truth.source=manual, you must provide ground_truth.texts")

    if gt_source == "file" and not config["ground_truth"].get("file"):
        raise ValueError("When using ground_truth.source=file, you must provide ground_truth.file")

    if gt_source == "file" and not os.path.isfile(config["ground_truth"].get("file")):
        raise ValueError(f"Ground truth file does not exist: {config['ground_truth'].get('file')}")


def override_config_with_cli_args(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """Override configuration with command-line arguments.

    Args:
        config: Dictionary containing configuration
        cli_args: Dictionary containing command-line arguments

    Returns:
        Updated configuration dictionary
    """
    # Deep copy the config to avoid modifying the original
    config = {**config}

    # Override input configuration
    if cli_args.get("image_paths"):
        config["input"]["image_paths"] = cli_args["image_paths"]
    if cli_args.get("image_folder"):
        config["input"]["image_folder"] = cli_args["image_folder"]
    if cli_args.get("image_patterns"):
        config["input"]["image_patterns"] = cli_args["image_patterns"]

    # Override model configuration
    if cli_args.get("custom_prompt"):
        config["models"]["custom_prompt"] = cli_args["custom_prompt"]

    # Override ground truth configuration
    if cli_args.get("ground_truth"):
        config["ground_truth"]["source"] = cli_args["ground_truth"]
    if cli_args.get("ground_truth_texts"):
        config["ground_truth"]["texts"] = cli_args["ground_truth_texts"]
    if cli_args.get("ground_truth_file"):
        config["ground_truth"]["file"] = cli_args["ground_truth_file"]

    # Override output configuration
    if cli_args.get("save_ground_truth"):
        config["output"]["ground_truth"]["save"] = cli_args["save_ground_truth"]
    if cli_args.get("ground_truth_dir"):
        config["output"]["ground_truth"]["directory"] = cli_args["ground_truth_dir"]
    if cli_args.get("save_ocr_results"):
        config["output"]["ocr_results"]["save"] = cli_args["save_ocr_results"]
    if cli_args.get("ocr_results_dir"):
        config["output"]["ocr_results"]["directory"] = cli_args["ocr_results_dir"]
    if cli_args.get("save_visualization"):
        config["output"]["visualization"]["save"] = cli_args["save_visualization"]
    if cli_args.get("visualization_dir"):
        config["output"]["visualization"]["directory"] = cli_args["visualization_dir"]

    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the configuration.

    Args:
        config: Dictionary containing configuration
    """
    print("\n===== OCR Comparison Pipeline Configuration =====")

    # Input configuration
    print("\nInput:")
    if config["input"].get("image_paths"):
        print(f"  • Using {len(config['input'].get('image_paths'))} specified image paths")
    if config["input"].get("image_folder"):
        print(f"  • Searching for images in folder: {config['input'].get('image_folder')}")
        print(f"  • Using patterns: {config['input'].get('image_patterns')}")

    # Model configuration
    print("\nModels:")
    if config["models"].get("custom_prompt"):
        print(f"  • Using custom prompt: {config['models'].get('custom_prompt')[:50]}...")
    else:
        print("  • Using default prompts")

    # Ground truth configuration
    print("\nGround Truth:")
    gt_source = config["ground_truth"].get("source", "none")
    print(f"  • Source: {gt_source}")
    if gt_source == "file":
        print(f"  • File: {config['ground_truth'].get('file')}")
    elif gt_source == "manual":
        print(f"  • Manual texts: {len(config['ground_truth'].get('texts', []))} provided")

    # Output configuration
    print("\nOutput:")
    if config["output"]["ground_truth"].get("save", False):
        print(f"  • Saving ground truth data to: {config['output']['ground_truth'].get('directory')}")
    if config["output"]["ocr_results"].get("save", False):
        print(f"  • Saving OCR results to: {config['output']['ocr_results'].get('directory')}")
    if config["output"]["visualization"].get("save", False):
        print(f"  • Saving visualization to: {config['output']['visualization'].get('directory')}")

    print("\n================================================\n")


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration.

    Returns:
        Dictionary containing default configuration
    """
    return {
        "input": {
            "image_paths": [],
            "image_folder": None,
            "image_patterns": ["*.jpg", "*.jpeg", "*.png", "*.webp"],
        },
        "models": {
            "custom_prompt": None,
        },
        "ground_truth": {
            "source": "none",
            "texts": [],
            "file": None,
        },
        "output": {
            "ground_truth": {
                "save": False,
                "directory": "ground_truth",
            },
            "ocr_results": {
                "save": False,
                "directory": "ocr_results",
            },
            "visualization": {
                "save": False,
                "directory": "visualizations",
            },
        },
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.

    Args:
        config: Dictionary containing configuration
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
