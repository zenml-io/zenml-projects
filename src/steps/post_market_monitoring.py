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

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict

from zenml import log_metadata, step

from src.constants import (
    DEPLOYMENT_INFO_NAME,
    EVALUATION_RESULTS_NAME,
    MONITORING_PLAN_NAME,
)


@step
def post_market_monitoring(
    deployment_info: Annotated[Dict[str, Any], DEPLOYMENT_INFO_NAME],
    evaluation_results: Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
) -> Annotated[Dict[str, Any], MONITORING_PLAN_NAME]:
    """Setup comprehensive post-market monitoring (Article 17).

    Creates a monitoring plan that satisfies EU AI Act Article 17 requirements
    for post-market monitoring of high-risk AI systems.

    Args:
        deployment_info: Information about the deployed model
        evaluation_results: Evaluation metrics and fairness analysis

    Returns:
        Monitoring plan details
    """
    timestamp = datetime.now().isoformat()

    # Define monitoring thresholds based on evaluation results
    baseline_accuracy = evaluation_results.get("metrics", {}).get("accuracy", 0.8)
    drift_threshold = 0.1  # Alert if metrics drop by more than 10%

    # Create comprehensive monitoring plan
    monitoring_plan = {
        "plan_id": f"monitoring_plan_{timestamp.replace(':', '-')}",
        "model_id": deployment_info.get("model_checksum", "unknown")[:8],
        "created_at": timestamp,
        "description": "Post-market monitoring plan for credit scoring model",
        "monitoring_frequency": {
            "data_drift": "daily",
            "performance_evaluation": "weekly",
            "fairness_audit": "monthly",
        },
        "alert_thresholds": {
            "accuracy_drop": baseline_accuracy - drift_threshold,
            "data_drift_score": 0.15,  # Example threshold for distribution shift
            "fairness_metrics": {"disparate_impact_min": 0.8, "disparate_impact_max": 1.25},
        },
        "response_procedures": {
            "minor_drift": "Log and monitor more frequently",
            "significant_drift": "Alert data science team and investigate",
            "critical_issue": "Trigger incident response and model reevaluation",
        },
        "responsible_parties": {
            "monitoring_owner": "data_science_team@example.com",
            "escalation_contact": "compliance_officer@example.com",
        },
    }

    # Save monitoring plan to file
    monitoring_dir = Path("compliance/monitoring")
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    with open(monitoring_dir / f"{monitoring_plan['plan_id']}.json", "w") as f:
        json.dump(monitoring_plan, f, indent=2)

    # Log monitoring plan metadata for compliance documentation
    log_metadata(metadata={f"monitoring_plan_{timestamp.replace(':', '-')}": monitoring_plan})

    print("✅ Post-market monitoring plan established (Article 17)")
    print("📈 Daily drift monitoring, weekly performance checks, monthly fairness audits")

    return monitoring_plan
