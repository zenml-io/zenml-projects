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

from src.constants import Artifacts as A
from src.constants import ModalConfig, Pipelines
from src.steps import (
    approve_deployment,
    generate_annex_iv_documentation,
    generate_compliance_dashboard,
    generate_sbom,
    modal_deployment,
    post_market_monitoring,
)


@pipeline(name=Pipelines.DEPLOYMENT)
def deployment(
    model: Annotated[Any, A.MODEL] = None,
    preprocess_pipeline: Annotated[Any, A.PREPROCESS_PIPELINE] = None,
    evaluation_results: Annotated[Any, A.EVALUATION_RESULTS] = None,
    risk_scores: Annotated[Any, A.RISK_SCORES] = None,
    environment: str = ModalConfig.ENVIRONMENT,
):
    """EU AI Act compliant deployment pipeline.

    Implements:
    - Article 14: Human oversight through approval process
    - Article 15: Accuracy & robustness via SBOM generation
    - Article 17: Post-market monitoring
    - Article 18: Incident reporting system

    Args:
        model: The trained model
        preprocess_pipeline: Preprocessing pipeline used in training
        evaluation_results: Model evaluation metrics and fairness analysis
        risk_scores: Risk assessment information
        environment: The environment to save the artifact to.

    Returns:
        Dictionary with deployment and monitoring information
    """
    # Fetch artifacts from ZenML if not provided
    client = Client()
    if model is None:
        model = client.get_artifact_version(name_id_or_prefix=A.MODEL)
    if evaluation_results is None:
        evaluation_results = client.get_artifact_version(
            name_id_or_prefix=A.EVALUATION_RESULTS
        )
    if risk_scores is None:
        risk_scores = client.get_artifact_version(
            name_id_or_prefix=A.RISK_SCORES
        )
    if preprocess_pipeline is None:
        preprocess_pipeline = client.get_artifact_version(
            name_id_or_prefix=A.PREPROCESS_PIPELINE
        )

    # Human oversight approval gate (Article 14)
    approved, approval_record = approve_deployment(
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
    )

    # Model deployment with integrated monitoring (Articles 10, 17, 18)
    deployment_info = modal_deployment(
        approved=approved,
        approval_record=approval_record,
        model=model,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
        environment=environment,
    )

    # Generate Software Bill of Materials for Article 15 (Accuracy & Robustness)
    generate_sbom(
        deployment_info=deployment_info,
    )

    # Post-market monitoring plan (Article 17)
    monitoring_plan = post_market_monitoring(
        deployment_info=deployment_info,
        evaluation_results=evaluation_results,
    )

    # Generate comprehensive technical documentation (Article 11)
    documentation_path, run_release_dir = generate_annex_iv_documentation(
        evaluation_results=evaluation_results,
        risk_scores=risk_scores,
        deployment_info=deployment_info,
    )

    # Generate compliance dashboard HTML visualization
    compliance_dashboard = generate_compliance_dashboard(
        run_release_dir=run_release_dir,
    )

    return (
        deployment_info,
        monitoring_plan,
        documentation_path,
        compliance_dashboard,
    )
