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

from typing import Any, Dict

from zenml.pipelines import pipeline

from steps import (
    approve_deployment,
    deploy_model,
    generate_annex_iv_documentation,
    post_market_monitoring,
)


@pipeline(enable_cache=False)
def deployment(
    model_path: str,
    evaluation_results: Dict[str, Any],
    risk_info: Dict[str, Any],
    preprocess_pipeline: Any,
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
    # Human oversight approval gate (Article 14)
    approved = approve_deployment(
        model_path=model_path, evaluation_results=evaluation_results, risk_info=risk_info
    )

    # Model deployment with integrated monitoring (Articles 10, 17, 18)
    deployment_info = deploy_model(
        model_path=model_path,
        approved=approved,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
    )

    # Comprehensive post-market monitoring plan (Article 17)
    monitoring_plan = post_market_monitoring(
        deployment_info=deployment_info,
        evaluation_results=evaluation_results,
    )

    docs_path = generate_annex_iv_documentation(
        model_path=model_path,
        evaluation_results=evaluation_results,
        risk_info=risk_info,
    )

    # Return deployment information
    return {
        "deployment": deployment_info,
        "monitoring": monitoring_plan,
        "approved": approved,
        "docs_path": docs_path,
    }
