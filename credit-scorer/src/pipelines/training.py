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
#

from typing import Any, Dict, List, Optional

from zenml import pipeline
from zenml.client import Client

from src.constants import (
    TEST_DATASET_NAME,
    TRAIN_DATASET_NAME,
    TRAINING_PIPELINE_NAME,
)
from src.steps import (
    evaluate_model,
    risk_assessment,
    train_model,
)


@pipeline(name=TRAINING_PIPELINE_NAME)
def training(
    train_df: Any = None,
    test_df: Any = None,
    target: str = "TARGET",
    hyperparameters: Optional[Dict[str, Any]] = None,
    protected_attributes: Optional[List[str]] = None,
    approval_thresholds: Optional[Dict[str, float]] = None,
    risk_register_path: str = "docs/risk/risk_register.xlsx",
    model_path: str = "models/model.pkl",
):
    """Training pipeline for credit scoring with EU AI Act compliance.

    This pipeline handles:
    1. Model training with design rationale (Article 11)
    2. Evaluation with fairness metrics (Articles 9, 15)
    3. Risk assessment (Article 9)

    Args:
        train_df: Training dataset.
        test_df: Test dataset.
        target: Name of the target column
        hyperparameters: Optional model hyperparameters
        protected_attributes: List of columns to check for fairness
        approval_thresholds: Dictionary containing approval thresholds
        risk_register_path: Path to the risk register
        model_path: Path to save the trained model

    Returns:
        model: The trained model
        eval_results: Evaluation metrics and fairness analysis
        risk_scores: Risk assessment results
    """
    # Retrieve datasets if not provided
    if train_df is None or test_df is None:
        client = Client()
        train_df = client.get_artifact_version(name_id_or_prefix=TRAIN_DATASET_NAME)
        test_df = client.get_artifact_version(name_id_or_prefix=TEST_DATASET_NAME)

    # Train model with provided data
    model = train_model(
        train_df=train_df,
        test_df=test_df,
        target=target,
        hyperparameters=hyperparameters,
        model_path=model_path,
    )

    # Evaluate model for performance and fairness
    eval_results = evaluate_model(
        test_df=test_df,
        protected_attributes=protected_attributes,
        target=target,
        model=model,
        approval_thresholds=approval_thresholds,
    )

    # Perform risk assessment based on evaluation results
    risk_scores = risk_assessment(
        evaluation_results=eval_results,
        risk_register_path=risk_register_path,
        approval_thresholds=approval_thresholds,
    )

    # Return artifacts to be used by deployment pipeline
    return model, eval_results, risk_scores
