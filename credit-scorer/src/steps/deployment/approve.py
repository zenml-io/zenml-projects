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

import os
from datetime import datetime
from typing import Annotated, Any, Dict, Tuple

from zenml import step

from src.constants import (
    APPROVAL_RECORD_NAME,
    APPROVED_NAME,
    EVALUATION_RESULTS_NAME,
    RISK_SCORES_NAME,
)


@step(enable_cache=False)
def approve_deployment(
    evaluation_results: Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
    risk_scores: Annotated[Dict[str, Any], RISK_SCORES_NAME],
    approval_thresholds: Dict[str, float],
) -> Tuple[
    Annotated[bool, APPROVED_NAME],
    Annotated[Dict[str, Any], APPROVAL_RECORD_NAME],
]:
    """Human oversight approval gate with comprehensive documentation (Article 14).

    Blocks deployment until a human reviews the model and approves it.
    Generates required documentation for EU AI Act Article 14 compliance.

    Args:
        evaluation_results: Dictionary containing evaluation metrics and fairness analysis
        risk_scores: Dictionary containing risk assessment information
        approval_thresholds: Dictionary containing approval thresholds for accuracy, bias disparity, and risk score

    Returns:
        Boolean indicating whether deployment is approved
    """
    # Timestamp for record-keeping
    timestamp = datetime.now().isoformat()

    # Create human-readable summary for the reviewer
    print("\n" + "=" * 50)
    print("  HUMAN OVERSIGHT REQUIRED (EU AI Act Article 14)  ")
    print("=" * 50)

    # Performance metrics summary
    print("\n📊 PERFORMANCE METRICS:")
    print(
        f"  • Accuracy: {evaluation_results['metrics'].get('accuracy', 'N/A'):.4f}"
    )
    print(f"  • AUC: {evaluation_results['metrics'].get('auc', 'N/A'):.4f}")

    # Fairness summary
    if "fairness_metrics" in evaluation_results:
        print("\n⚖️ FAIRNESS ASSESSMENT:")
        for attribute, metrics in evaluation_results[
            "fairness_metrics"
        ].items():
            print(
                f"  • {attribute.capitalize()} disparate impact: {metrics.get('disparate_impact', 'N/A'):.2f}"
            )
            print(
                f"  • {attribute.capitalize()} demographic parity: {metrics.get('demographic_parity', 'N/A'):.4f}"
            )

    # Risk assessment summary
    print("\n⚠️ RISK ASSESSMENT:")
    print(f"  • Risk level: {risk_scores.get('risk_level', 'N/A')}")

    if "high_risk_factors" in risk_scores and risk_scores["high_risk_factors"]:
        print("  • High risk factors detected:")
        for factor in risk_scores["high_risk_factors"]:
            print(f"    - {factor}")

    if (
        "mitigation_measures" in risk_scores
        and risk_scores["mitigation_measures"]
    ):
        print("  • Mitigation measures:")
        for measure in risk_scores["mitigation_measures"]:
            print(f"    - {measure}")

    # Create threshold checks
    threshold_checks = {
        "Accuracy": evaluation_results["metrics"].get("accuracy", 0)
        >= approval_thresholds["accuracy"],
        "Bias disparity": all(
            metrics.get("selection_rate_disparity", 1)
            <= approval_thresholds["bias_disparity"]
            for attr, metrics in evaluation_results.get(
                "fairness_metrics", {}
            ).items()
        ),
        "Risk score": risk_scores.get("overall", 1)
        <= approval_thresholds["risk_score"],
    }

    # Display threshold check results
    print("\n🔍 THRESHOLD CHECKS:")
    for check_name, passed in threshold_checks.items():
        status = "✅ PASS" if passed else "⚠️ FAIL"
        print(f"  • {check_name}: {status}")

    # Decision prompt
    print("\n📝 APPROVAL DECISION:")

    # Check for automated decision via environment variable (e.g., in CI pipeline)
    decision = os.getenv("DEPLOY_APPROVAL")

    if decision is None:
        # Interactive mode - request human input
        decision = input("\nApprove deployment? (y/N): ").strip().lower()
        approver = os.getenv("USER", input("Approver name: ").strip())
        rationale = input("Decision rationale: ").strip()
        decision_mode = "interactive"
    else:
        # Automated mode
        approver = os.getenv("APPROVER", "automated")
        rationale = os.getenv(
            "APPROVAL_RATIONALE", "Automated approval via environment variable"
        )
        decision_mode = "automated"

    approved = decision == "y"

    # Create documented record for compliance
    approval_record = {
        "approval_id": f"approval_{timestamp.replace(':', '-')}",
        "timestamp": timestamp,
        "approved": approved,
        "approver": approver,
        "rationale": rationale,
        "decision_mode": decision_mode,
        "threshold_checks": {
            check: passed for check, passed in threshold_checks.items()
        },
        "evaluation_summary": {
            "accuracy": evaluation_results["metrics"].get("accuracy", None),
            "auc": evaluation_results["metrics"].get("auc", None),
            "fairness_flags": evaluation_results.get("fairness_flags", []),
        },
        "risk_summary": {
            "risk_level": risk_scores.get("risk_level", "unknown"),
            "high_risk_factors": risk_scores.get("high_risk_factors", []),
        },
    }

    # Final decision message
    if approved:
        print("\n✅ DEPLOYMENT APPROVED")
    else:
        print("\n❌ DEPLOYMENT REJECTED")
        raise RuntimeError(f"Deployment rejected by {approver}: {rationale}")

    return approved, approval_record
