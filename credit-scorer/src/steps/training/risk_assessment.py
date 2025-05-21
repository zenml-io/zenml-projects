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

from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List

from openpyxl import Workbook, load_workbook
from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger

from src.constants import (
    HAZARD_DEFINITIONS,
    RISK_SCORES_NAME,
)
from src.utils.storage import save_artifact_to_modal

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
RiskScores = Annotated[Dict[str, float], RISK_SCORES_NAME]
# --------------------------------------------------------------------------- #


def identify_hazards(evaluation_results: Dict, scores: Dict) -> List[Dict]:
    """Identify applicable hazards based on evaluation results."""
    identified_hazards = []

    # Evaluate each hazard trigger function
    for hazard_id, hazard_info in HAZARD_DEFINITIONS.items():
        try:
            trigger_func = hazard_info["trigger"]
            if trigger_func(evaluation_results, scores):
                identified_hazards.append(
                    {
                        "id": hazard_id,
                        "description": hazard_info["description"],
                        "severity": hazard_info["severity"],
                        "mitigation": hazard_info["mitigation"],
                    }
                )
        except (KeyError, TypeError) as e:
            logger.warning(f"Could not evaluate hazard {hazard_id}: {e}")

    return identified_hazards


def score_risk(evaluation: Dict) -> Dict[str, float]:
    """Return dict with per-factor risk ∈ [0,1] + overall."""
    # Get metrics with default fallbacks
    metrics = evaluation.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    # Get AUC with default
    auc = metrics.get("auc", 0.5)

    # Get fairness info with defaults
    fairness = evaluation.get("fairness", {})
    if not isinstance(fairness, dict):
        fairness = {}

    bias_flag = fairness.get("bias_flag", False)

    # Calculate max disparity across groups
    disparities = []
    fairness_metrics = fairness.get("fairness_metrics", {})
    if isinstance(fairness_metrics, dict):
        for group_metrics in fairness_metrics.values():
            if isinstance(group_metrics, dict) and "selection_rate_disparity" in group_metrics:
                disparities.append(abs(group_metrics["selection_rate_disparity"]))

    disparity = max(disparities) if disparities else 0.0

    # Calculate risk scores
    risk_auc = 1 - auc  # low AUC → higher risk
    risk_bias = 0.8 if bias_flag else disparity  # flat score if flagged
    overall = round(min(1.0, 0.5 * risk_auc + 0.5 * risk_bias), 3)

    return {
        "risk_auc": round(risk_auc, 3),
        "risk_bias": round(risk_bias, 3),
        "overall": overall,
    }


def get_status(risk_score: float, approval_thresholds: Dict[str, float]) -> str:
    """Determine row status for the risk register.

    Args:
        risk_score: The overall risk score ∈ [0,1].
        approval_thresholds: Dictionary containing approval thresholds.

    Returns:
        "PENDING" if the score exceeds the configured threshold, otherwise "COMPLETED".
    """
    threshold = approval_thresholds["risk_score"]
    return "PENDING" if risk_score > threshold else "COMPLETED"


def get_article_for_hazard(hazard_id: str) -> str:
    """Map hazards to specific EU AI Act articles."""
    article_mapping = {
        "bias_protected_groups": "article_10",  # Data and Data Governance
        "poor_performance": "article_15",  # Accuracy, Robustness and Cybersecurity
        "data_quality_issues": "article_10",  # Data and Data Governance
        "model_drift": "article_17",  # Post-market Monitoring
        "security_vulnerability": "article_15",  # Accuracy, Robustness and Cybersecurity
        "transparency_issues": "article_13",  # Transparency and Provision of Information
        "human_oversight_failure": "article_14",  # Human Oversight
        "documentation_gaps": "article_11",  # Technical Documentation
    }
    return article_mapping.get(hazard_id, "article_9")  # Default to Risk Management


@step
def risk_assessment(
    evaluation_results: Dict,
    approval_thresholds: Dict[str, float],
    risk_register_path: str = "docs/risk/risk_register.xlsx",
) -> RiskScores:
    """Compute risk scores & update register. Article 9 compliant."""
    scores = score_risk(evaluation_results)
    hazards = identify_hazards(evaluation_results, scores)

    headers = [
        "Run_ID",
        "Timestamp",
        "Risk_overall",
        "Risk_auc",
        "Risk_bias",
        "Risk_category",
        "Status",
        "Risk_description",
        "Mitigation",
        "Article",
        "Mitigation_status",
        "Review_date",
    ]

    # Load or create workbook
    wb_path = Path(risk_register_path)
    if wb_path.exists():
        wb = load_workbook(wb_path)
        if "Risks" in wb.sheetnames:
            ws = wb["Risks"]
        else:
            ws = wb.create_sheet("Risks")
            ws.append(headers)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "Risks"
        ws.append(headers)

    run_id = get_step_context().pipeline_run.id
    timestamp = datetime.now().isoformat()

    # Process hazards
    if hazards:
        for hz in hazards:
            status = get_status(scores["overall"], approval_thresholds)
            article = get_article_for_hazard(hz["id"])

            ws.append(
                [
                    str(run_id),
                    timestamp,
                    scores["overall"],
                    scores["risk_auc"],
                    scores["risk_bias"],
                    hz["severity"].upper(),
                    status,
                    hz["description"],
                    hz["mitigation"],
                    article,
                    status,
                    timestamp,
                ]
            )

        # HazardDetails sheet
        if "HazardDetails" not in wb.sheetnames:
            hazard_sheet = wb.create_sheet("HazardDetails")
            hazard_sheet.append(
                [
                    "Run_ID",
                    "Timestamp",
                    "Risk_Score",
                    "Hazard_ID",
                    "Description",
                    "Severity",
                    "Mitigation",
                    "Details",
                    "Article",
                ]
            )
        else:
            hazard_sheet = wb["HazardDetails"]

        for hz in hazards:
            details = ""
            if hz["id"] == "bias_protected_groups" and "fairness" in evaluation_results:
                fairness = evaluation_results["fairness"]
                if isinstance(fairness, dict) and "fairness_metrics" in fairness:
                    for attr, metrics in fairness.get("fairness_metrics", {}).items():
                        if isinstance(metrics, dict) and "selection_rate_disparity" in metrics:
                            disparity = metrics["selection_rate_disparity"]
                            if abs(disparity) > 0.2:
                                details += f"{attr}: {abs(disparity):.3f} disparity\n"

            article = get_article_for_hazard(hz["id"])
            hazard_sheet.append(
                [
                    str(run_id),
                    timestamp,
                    scores["overall"],
                    hz["id"],
                    hz["description"],
                    hz["severity"].upper(),
                    hz["mitigation"],
                    details,
                    article,
                ]
            )
    else:
        # No hazards case
        status = get_status(scores["overall"], approval_thresholds)
        ws.append(
            [
                str(run_id),
                timestamp,
                scores["overall"],
                scores["risk_auc"],
                scores["risk_bias"],
                "LOW",
                status,
                "No hazards identified",
                "N/A",
                "article_9",
                status,
                timestamp,
            ]
        )

    wb.save(wb_path)
    save_artifact_to_modal(artifact=wb, artifact_path=risk_register_path)

    result = {
        **scores,
        "hazards": hazards,
        "risk_register_path": str(risk_register_path),
    }
    log_metadata(metadata=result)
    return result
