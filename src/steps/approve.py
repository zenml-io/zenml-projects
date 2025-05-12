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
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict

from zenml import log_metadata, step

from src.constants import (
    APPROVED_NAME,
    EVALUATION_RESULTS_NAME,
    RISK_SCORES_NAME,
)


@step(enable_cache=False)
def approve_deployment(
    volume_metadata: Annotated[Dict, "volume_metadata"],
    evaluation_results: Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
    risk_scores: Annotated[Dict[str, Any], RISK_SCORES_NAME],
) -> Annotated[bool, APPROVED_NAME]:
    """Human oversight approval gate with comprehensive documentation (Article 14).

    Blocks deployment until a human reviews the model and approves it.
    Generates required documentation for EU AI Act Article 14 compliance.

    Args:
        volume_metadata: Metadata for the Modal Volume
        evaluation_results: Dictionary containing evaluation metrics and fairness analysis
        risk_scores: Dictionary containing risk assessment information

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
    print(f"  • Accuracy: {evaluation_results['metrics'].get('accuracy', 'N/A'):.4f}")
    print(f"  • AUC: {evaluation_results['metrics'].get('auc', 'N/A'):.4f}")
    f1_score = evaluation_results.get("metrics", {}).get("f1")
    if f1_score is not None:
        print(f"  • F1 Score: {f1_score:.4f}")
    else:
        print("  • F1 Score: N/A")

    # Fairness summary
    if "fairness_metrics" in evaluation_results:
        print("\n⚖️ FAIRNESS ASSESSMENT:")
        for attribute, metrics in evaluation_results["fairness_metrics"].items():
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

    if "mitigation_measures" in risk_scores and risk_scores["mitigation_measures"]:
        print("  • Mitigation measures:")
        for measure in risk_scores["mitigation_measures"]:
            print(f"    - {measure}")

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
        rationale = os.getenv("APPROVAL_RATIONALE", "Automated approval via environment variable")
        decision_mode = "automated"

    approved = decision == "y"

    # Create documented record for compliance
    approval_record = {
        "approval_id": f"approval_{timestamp.replace(':', '-')}",
        "timestamp": timestamp,
        "volume_metadata": volume_metadata,
        "approved": approved,
        "approver": approver,
        "rationale": rationale,
        "decision_mode": decision_mode,
        "evaluation_summary": {
            "accuracy": evaluation_results["metrics"].get("accuracy", None),
            "auc": evaluation_results["metrics"].get("auc", None),
            "f1": evaluation_results["metrics"].get("f1", None),
            "fairness_flags": evaluation_results.get("fairness_flags", []),
        },
        "risk_summary": {
            "risk_level": risk_scores.get("risk_level", "unknown"),
            "high_risk_factors": risk_scores.get("high_risk_factors", []),
        },
    }

    # Save approval record to file
    approval_dir = Path("compliance/approval_records")
    approval_dir.mkdir(parents=True, exist_ok=True)

    with open(approval_dir / f"approval_{timestamp.replace(':', '-')}.json", "w") as f:
        json.dump(approval_record, f, indent=2)

    # Log approval metadata for Annex IV documentation
    log_metadata(metadata={"approval_record": approval_record})

    # Final decision message
    if approved:
        print("\n✅ DEPLOYMENT APPROVED")
    else:
        print("\n❌ DEPLOYMENT REJECTED")
        raise RuntimeError(f"Deployment rejected by {approver}: {rationale}")

    return approved
