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

from src.constants import Artifacts as A


@step(enable_cache=False)
def approve_deployment(
    evaluation_results: Annotated[Dict[str, Any], A.EVALUATION_RESULTS],
    risk_scores: Annotated[Dict[str, Any], A.RISK_SCORES],
    approval_thresholds: Dict[str, float],
) -> Tuple[
    bool,
    Annotated[Dict[str, Any], A.APPROVAL_RECORD],
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

    print("\n" + "=" * 60)
    print("  HUMAN OVERSIGHT REQUIRED (EU AI Act Article 14)  ")
    print("=" * 60)

    # Extract metrics for display
    metrics = evaluation_results.get("metrics", {})
    fairness_data = evaluation_results.get("fairness", {})
    fairness_metrics = fairness_data.get("fairness_metrics", {})
    bias_flag = fairness_data.get("bias_flag", False)

    # Performance metrics summary
    print("\n📊 PERFORMANCE METRICS:")
    print(f"  • Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  • Precision: {metrics.get('precision', 'N/A'):.4f}")
    print(f"  • Recall: {metrics.get('recall', 'N/A'):.4f}")
    print(f"  • F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
    print(f"  • AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
    print(
        f"  • Average Precision: {metrics.get('average_precision', 'N/A'):.4f}"
    )
    print(
        f"  • Balanced Accuracy: {metrics.get('balanced_accuracy', 'N/A'):.4f}"
    )

    # Financial impact metrics
    print("\n💰 FINANCIAL IMPACT:")
    print(
        f"  • Optimal Threshold: {metrics.get('optimal_threshold', 'N/A'):.4f}"
    )
    print(f"  • Normalized Cost: {metrics.get('normalized_cost', 'N/A'):.4f}")

    # Fairness summary (aggregated, not per-group)
    print(f"\n⚖️ FAIRNESS ASSESSMENT:")
    if bias_flag:
        print("  🚨 BIAS DETECTED - Requires careful review")

        # Show worst disparity without listing all groups
        max_disparity = 0
        worst_attribute = None

        for attribute, attr_metrics in fairness_metrics.items():
            disparity = abs(attr_metrics.get("selection_rate_disparity", 0))
            if disparity > max_disparity:
                max_disparity = disparity
                worst_attribute = attribute

        if worst_attribute:
            print(
                f"  • Highest disparity: {max_disparity:.3f} ({worst_attribute})"
            )

        print(f"  • Protected attributes analyzed: {len(fairness_metrics)}")
    else:
        print("  ✅ No significant bias detected across protected groups")
        print(f"  • Protected attributes analyzed: {len(fairness_metrics)}")

    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print(f"  • Overall Risk Score: {risk_scores.get('overall', 0):.3f}")
    print(f"  • Risk Level: {risk_scores.get('risk_level', 'Unknown')}")

    high_risk_count = len(risk_scores.get("high_risk_factors", []))
    if high_risk_count > 0:
        print(f"  • High-risk factors identified: {high_risk_count}")

    # Approval criteria check
    threshold_checks = {
        "Performance": metrics.get("accuracy", 0)
        >= approval_thresholds.get("accuracy", 0.7),
        "Fairness": not bias_flag,
        "Risk": risk_scores.get("overall", 1)
        <= approval_thresholds.get("risk_score", 0.8),
    }

    print(f"\n🔍 APPROVAL CRITERIA:")
    all_passed = True
    for check_name, passed in threshold_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  • {check_name}: {status}")
        if not passed:
            all_passed = False

    print(
        f"\n📝 RECOMMENDATION: {'✅ APPROVE' if all_passed else '⚠️ REVIEW REQUIRED'}"
    )

    # Get decision
    decision = os.getenv("DEPLOY_APPROVAL")

    if decision is None:
        if bias_flag:
            print("\n⚠️ WARNING: Review fairness implications before approval")

        decision = (
            input(
                f"\nApprove deployment? ({'Y/n' if all_passed else 'y/N'}): "
            )
            .strip()
            .lower()
        )
        approver = os.getenv("USER", input("Approver name: ").strip())
        rationale = input("Decision rationale: ").strip()
        decision_mode = "interactive"
    else:
        approver = os.getenv("APPROVER", "automated")
        rationale = os.getenv("APPROVAL_RATIONALE", "Automated approval")
        decision_mode = "automated"

    # Handle default approval logic
    if decision == "":
        approved = all_passed  # Default to approve only if all criteria pass
    else:
        approved = decision in ["y", "yes"]

    # Create approval record
    approval_record = {
        "approval_id": f"approval_{timestamp.replace(':', '-')}",
        "timestamp": timestamp,
        "approved": approved,
        "approver": approver,
        "rationale": rationale,
        "decision_mode": decision_mode,
        "criteria_met": all_passed,
        "bias_detected": bias_flag,
        "key_metrics": {
            "accuracy": metrics.get("accuracy"),
            "f1_score": metrics.get("f1_score"),
            "auc_roc": metrics.get("auc_roc"),
            "cost_per_application": metrics.get("normalized_cost"),
            "risk_score": risk_scores.get("overall"),
        },
        "protected_attributes_count": len(fairness_metrics),
        "max_bias_disparity": max(
            [
                abs(attr_metrics.get("selection_rate_disparity", 0))
                for attr_metrics in fairness_metrics.values()
            ]
        )
        if fairness_metrics
        else 0,
    }

    # Final status
    if approved:
        print(f"\n✅ DEPLOYMENT APPROVED by {approver}")
    else:
        print(f"\n❌ DEPLOYMENT REJECTED by {approver}")
        raise RuntimeError(f"Deployment rejected by {approver}: {rationale}")

    return approved, approval_record
