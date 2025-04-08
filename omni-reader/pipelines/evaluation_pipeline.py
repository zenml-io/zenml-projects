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

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.logger import get_logger

from steps import (
    evaluate_models,
    load_ground_truth_texts,
    load_ocr_results,
)

load_dotenv()

logger = get_logger(__name__)

docker_settings = DockerSettings(
    requirements="requirements.txt",
    required_integrations=["s3", "aws"],
    python_package_installer="uv",
    environment={
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
    },
)


@pipeline(settings={"docker": docker_settings})
def ocr_evaluation_pipeline(
    ground_truth_folder: Optional[str] = None,
    ground_truth_files: Optional[List[str]] = None,
) -> None:
    """Run OCR evaluation pipeline comparing existing model results."""
    if not ground_truth_folder and not ground_truth_files:
        raise ValueError(
            "Either ground_truth_folder or ground_truth_files must be provided for evaluation"
        )

    model_results = load_ocr_results(artifact_name="ocr_results")

    ground_truth_df = load_ground_truth_texts(
        model_results=model_results,
        ground_truth_folder=ground_truth_folder,
        ground_truth_files=ground_truth_files,
    )

    evaluate_models(
        model_results=model_results,
        ground_truth_df=ground_truth_df,
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

    pipeline_instance = ocr_evaluation_pipeline.with_options(
        enable_artifact_metadata=config.get("enable_artifact_metadata", True),
        enable_artifact_visualization=config.get("enable_artifact_visualization", True),
        enable_cache=config.get("enable_cache", False),
        enable_step_logs=config.get("enable_step_logs", True),
    )

    load_ground_truth_texts_params = (
        config.get("steps", {}).get("load_ground_truth_texts", {}).get("parameters", {})
    )

    pipeline_instance(
        ground_truth_folder=load_ground_truth_texts_params.get("ground_truth_folder"),
        ground_truth_files=load_ground_truth_texts_params.get("ground_truth_files", []),
    )
