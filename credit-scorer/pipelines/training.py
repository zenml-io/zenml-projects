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

import pandas as pd
from zenml.pipelines import pipeline

from steps import (
    evaluate_model,
    risk_assessment,
    train_model,
)


@pipeline
def training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "loan_approved",
    hyperparameters: Optional[Dict[str, Any]] = None,
    protected_attributes: Optional[List[str]] = None,
):
    """Training pipeline for credit scoring with EU AI Act compliance.

    This pipeline handles:
    1. Model training with design rationale (Article 11)
    2. Evaluation with fairness metrics (Articles 9, 15)
    3. Risk assessment (Article 9)

    Args:
        train_df: Training data from feature_engineering pipeline
        test_df: Test data from feature_engineering pipeline
        target: Name of the target column
        hyperparameters: Optional model hyperparameters
        protected_attributes: List of columns to check for fairness

    Returns:
        Dictionary with model, evaluation results, and risk assessment
    """
    # Set defaults for protected attributes
    if protected_attributes is None:
        protected_attributes = ["gender", "age_group"]

    # Train model with provided data
    model = train_model(
        train_df=train_df,
        test_df=test_df,  # Use test_df as validation set
        target=target,
        hyperparameters=hyperparameters,
    )

    # Evaluate model for performance and fairness
    eval_results = evaluate_model(
        model=model,
        test_df=test_df,
        target=target,
        protected_attributes=protected_attributes,
    )

    # Perform risk assessment based on evaluation results
    risk_info = risk_assessment(evaluation_results=eval_results)

    # Return artifacts to be used by deployment pipeline
    return {
        "model_path": model,
        "evaluation": eval_results,
        "risk": risk_info,
    }
