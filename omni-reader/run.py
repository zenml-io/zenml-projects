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
"""Run OCR comparison pipeline using ZenML with YAML configuration."""

import argparse
import os

from pipelines.ocr_pipeline import run_ocr_pipeline
from utils.config import (
    create_default_config,
    load_config,
    override_config_with_cli_args,
    print_config_summary,
    save_config,
    validate_config,
)
from utils.io_utils import list_available_ground_truth_files


def main():
    """Run the OCR comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Run OCR comparison between Mistral and Gemma3 using ZenML"
    )

    # Config file options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default="configs/ocr_config.yaml",
        help="Path to YAML configuration file",
    )
    config_group.add_argument(
        "--create-default-config",
        type=str,
        metavar="PATH",
        help="Create a default configuration file at the specified path and exit",
    )

    # Quick access options
    input_group = parser.add_argument_group("Input options (override config)")
    input_group.add_argument(
        "--image-paths",
        nargs="+",
        help="Paths to images to process (overrides config)",
    )
    input_group.add_argument(
        "--image-folder",
        type=str,
        help="Folder containing images to process (overrides config)",
    )

    # Ground truth utilities
    gt_group = parser.add_argument_group("Ground truth utilities")
    gt_group.add_argument(
        "--list-ground-truth-files",
        action="store_true",
        help="List available ground truth files and exit",
    )
    gt_group.add_argument(
        "--ground-truth-dir",
        type=str,
        default="ocr_results",
        help="Directory to look for ground truth files (for --list-ground-truth-files)",
    )

    args = parser.parse_args()

    # Create default config if requested
    if args.create_default_config:
        default_config = create_default_config()
        save_config(default_config, args.create_default_config)
        print(f"Default configuration saved to: {args.create_default_config}")
        return

    # Check if we should just list available ground truth files
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

        # Convert argparse Namespace to dict for overriding
        cli_args = {k: v for k, v in vars(args).items() if v is not None}

        # Override config with CLI arguments
        config = override_config_with_cli_args(config, cli_args)
    else:
        parser.error("Please provide a configuration file with --config")

    # Validate configuration
    try:
        validate_config(config)
    except ValueError as e:
        parser.error(str(e))

    # Print configuration summary
    print_config_summary(config)

    # Create output directories
    if config["output"]["ground_truth"].get("save", False):
        os.makedirs(config["output"]["ground_truth"]["directory"], exist_ok=True)
    if config["output"]["ocr_results"].get("save", False):
        os.makedirs(config["output"]["ocr_results"]["directory"], exist_ok=True)
    if config["output"]["visualization"].get("save", False):
        os.makedirs(config["output"]["visualization"]["directory"], exist_ok=True)

    run_ocr_pipeline(config)


if __name__ == "__main__":
    main()
