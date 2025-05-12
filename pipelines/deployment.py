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

from typing import Annotated, Any, Dict

from zenml.client import Client
from zenml.pipelines import pipeline

from constants import (
    EVALUATION_RESULTS_NAME,
    MODEL_PATH,
    PREPROCESS_PIPELINE_NAME,
    RISK_SCORES_NAME,
)
from steps import (
    approve_deployment,
    deploy_model,
    generate_annex_iv_documentation,
    post_market_monitoring,
)


@pipeline(enable_cache=False)
def deployment(
    model_path: Annotated[str, MODEL_PATH],
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
        model_path: Path to the trained model
        evaluation_results: Model evaluation metrics and fairness analysis
        risk_info: Risk assessment information
        preprocess_pipeline: Preprocessing pipeline used in training

    Returns:
        Dictionary with deployment and monitoring information
    """
    client = Client()
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
        model_path=model_path,
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
    )

    # Model deployment with integrated monitoring (Articles 10, 17, 18)
    deployment_info = deploy_model(
        model_path=model_path,
        approved=approved,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
    )

    # Comprehensive post-market monitoring plan (Article 17)
    post_market_monitoring(
        deployment_info=deployment_info,
        evaluation_results=evaluation_results,
    )

    generate_annex_iv_documentation(
        model_path=model_path,
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
    )
