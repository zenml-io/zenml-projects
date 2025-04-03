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
"""OCR Evaluation Pipeline implementation for comparing models using existing results."""

from typing import Any, Dict, List, Optional

import polars as pl
from dotenv import load_dotenv
from zenml import pipeline, step
from zenml.logger import get_logger

from steps import (
    evaluate_models,
    load_ground_truth_texts,
    load_ocr_results,
    save_visualization,
)

load_dotenv()

logger = get_logger(__name__)


@pipeline
def ocr_evaluation_pipeline(
    model_names: List[str] = None,
    results_dir: str = "ocr_results",
    result_files: Optional[List[str]] = None,
    ground_truth_folder: Optional[str] = None,
    ground_truth_files: Optional[List[str]] = None,
    save_visualization_data: bool = False,
    visualization_output_dir: str = "visualizations",
) -> None:
    """Run OCR evaluation pipeline comparing existing model results."""
    if not model_names or len(model_names) < 2:
        raise ValueError("At least two models are required for comparison")

    if not ground_truth_folder and not ground_truth_files:
        raise ValueError(
            "Either ground_truth_folder or ground_truth_files must be provided for evaluation"
        )

    model_results = load_ocr_results(
        model_names=model_names,
        results_dir=results_dir,
        result_files=result_files,
    )

    ground_truth_df = load_ground_truth_texts(
        model_results=model_results,
        ground_truth_folder=ground_truth_folder,
        ground_truth_files=ground_truth_files,
    )

    visualization = evaluate_models(
        model_results=model_results,
        ground_truth_df=ground_truth_df,
    )

    if save_visualization_data:
        save_visualization(
            visualization,
            output_dir=visualization_output_dir,
        )


def run_ocr_evaluation_pipeline(config: Dict[str, Any]) -> None:
    """Run the OCR evaluation pipeline from a configuration dictionary.

    Args:
        config: Dictionary containing configuration

    Returns:
        None
    """
    mode = config.get("parameters", {}).get("mode", "evaluation")
    if mode != "evaluation":
        logger.warning(f"Expected mode 'evaluation', but got '{mode}'. Proceeding anyway.")

    selected_models = config.get("parameters", {}).get("selected_models", [])
    if len(selected_models) < 2:
        raise ValueError("At least two models are required for evaluation")

    model_registry = config.get("models_registry", [])
    if not model_registry:
        raise ValueError("models_registry is required in the config")

    # Get model names from registry by using the passed models (may be shorthands or full names)
    model_names = []
    shorthand_to_name = {
        m.get("shorthand"): m.get("name") for m in model_registry if "shorthand" in m
    }

    for model_id in selected_models:
        if model_id in shorthand_to_name:
            model_names.append(shorthand_to_name[model_id])
        else:
            if any(m.get("name") == model_id for m in model_registry):
                model_names.append(model_id)
            else:
                logger.warning(f"Model '{model_id}' not found in registry, using as-is")
                model_names.append(model_id)

    if len(selected_models) < 2:
        raise ValueError("At least two models are required for evaluation")

    # Set up pipeline options
    pipeline_instance = ocr_evaluation_pipeline.with_options(
        enable_cache=config.get("enable_cache", False),
        enable_artifact_metadata=config.get("enable_artifact_metadata", True),
        enable_artifact_visualization=config.get("enable_artifact_visualization", True),
    )

    evaluate_models_params = (
        config.get("steps", {}).get("evaluate_models", {}).get("parameters", {})
    )
    save_visualization_params = (
        config.get("steps", {}).get("save_visualization", {}).get("parameters", {})
    )

    pipeline_instance(
        model_names=model_names,
        results_dir=evaluate_models_params.get("results_dir", "ocr_results"),
        result_files=evaluate_models_params.get("result_files"),
        ground_truth_folder=evaluate_models_params.get("ground_truth_folder"),
        ground_truth_files=evaluate_models_params.get("ground_truth_files", []),
        save_visualization_data=save_visualization_params.get("save_locally", False),
        visualization_output_dir=save_visualization_params.get("output_dir", "visualizations"),
    )
