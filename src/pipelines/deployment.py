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

from typing import Annotated, Any

from zenml.client import Client
from zenml.pipelines import pipeline

from src.constants import (
    DEPLOYMENT_PIPELINE_NAME,
    EVALUATION_RESULTS_NAME,
    MODEL_NAME,
    PREPROCESS_PIPELINE_NAME,
    RISK_SCORES_NAME,
)
from src.steps import (
    approve_deployment,
    generate_annex_iv_documentation,
    modal_deployment,
    post_market_monitoring,
)


@pipeline(name=DEPLOYMENT_PIPELINE_NAME)
def deployment(
    model: Annotated[Any, MODEL_NAME] = None,
    preprocess_pipeline: Annotated[Any, PREPROCESS_PIPELINE_NAME] = None,
    evaluation_results: Annotated[Any, EVALUATION_RESULTS_NAME] = None,
    risk_scores: Annotated[Any, RISK_SCORES_NAME] = None,
):
    """EU AI Act compliant deployment pipeline.

    Implements:
    - Article 14: Human oversight through approval process
    - Article 17: Post-market monitoring
    - Article 18: Incident reporting system

    Args:
        model: The trained model
        preprocess_pipeline: Preprocessing pipeline used in training
        evaluation_results: Model evaluation metrics and fairness analysis
        risk_scores: Risk assessment information

    Returns:
        Dictionary with deployment and monitoring information
    """
    # Fetch artifacts from ZenML if not provided
    client = Client()
    if model is None:
        model = client.get_artifact_version(name_id_or_prefix=MODEL_NAME)
    if evaluation_results is None:
        evaluation_results = client.get_artifact_version(name_id_or_prefix=EVALUATION_RESULTS_NAME)
    if risk_scores is None:
        risk_scores = client.get_artifact_version(name_id_or_prefix=RISK_SCORES_NAME)
    if preprocess_pipeline is None:
        preprocess_pipeline = client.get_artifact_version(
            name_id_or_prefix=PREPROCESS_PIPELINE_NAME
        )

    # Human oversight approval gate (Article 14)
    approved = approve_deployment(
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
    )

    # Model deployment with integrated monitoring (Articles 10, 17, 18)
    deployment_info = modal_deployment(
        approved=approved,
        model=model,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
    )

    # Post-market monitoring plan (Article 17)
    post_market_monitoring(
        deployment_info=deployment_info,
        evaluation_results=evaluation_results,
    )

    generate_annex_iv_documentation(
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
    )
