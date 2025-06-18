# Apache Software License 2.0
# Copyright (c) ZenML GmbH 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import time
from datetime import datetime
from typing import Annotated, Any, Dict, Tuple

from zenml import get_step_context, step
from zenml.client import Client
from zenml.integrations.slack.alerters.slack_alerter import (
    SlackAlerterParameters,
    SlackAlerterPayload,
)

from src.constants import Artifacts as A
from src.constants.config import SlackConfig as SC


@step(
    enable_cache=False,
    settings={"alerter": {"slack_channel_id": SC.CHANNEL_ID}},
)
def approve_deployment(
    evaluation_results: Annotated[Dict[str, Any], A.EVALUATION_RESULTS],
    risk_scores: Annotated[Dict[str, Any], A.RISK_SCORES],
    approval_thresholds: Dict[str, float],
) -> Tuple[
    Annotated[bool, A.APPROVED], Annotated[Dict[str, Any], A.APPROVAL_RECORD]
]:
    """Human oversight approval gate with comprehensive documentation (Article 14)."""
    # Extract context and metrics
    ctx = get_step_context()
    client = Client()
    pipeline_name, step_name, stack_name, run_id = (
        ctx.pipeline.name,
        ctx.step_run.name,
        client.active_stack.name,
        str(ctx.pipeline_run.id),
    )

    metrics = evaluation_results.get("metrics", {})
    fairness_data = evaluation_results.get("fairness", {})
    fairness_metrics = fairness_data.get("fairness_metrics", {})

    accuracy, f1_score, auc_roc = (
        metrics.get("accuracy", 0),
        metrics.get("f1_score", 0),
        metrics.get("auc_roc", 0),
    )
    risk_score = risk_scores.get("overall", 1)
    bias_flag = fairness_data.get("bias_flag", False)

    # Calculate minimum disparate impact ratio across all protected attributes
    # Lower DI ratios indicate worse bias (DI < 0.8 is adverse impact)
    min_di_ratio = (
        min(
            [
                attr.get("disparate_impact_ratio", 1.0)
                for attr in fairness_metrics.values()
            ]
        )
        if fairness_metrics
        else 1.0
    )

    # Check approval criteria
    perf_ok = accuracy >= approval_thresholds.get("accuracy", 0.7)
    # DI ratio >= 0.8 means no adverse impact
    di_threshold = approval_thresholds.get("disparate_impact_threshold", 0.8)
    fairness_ok = not bias_flag and min_di_ratio >= di_threshold
    risk_ok = risk_score <= approval_thresholds.get("risk_score", 0.4)
    all_ok = perf_ok and fairness_ok and risk_ok

    def create_blocks(header, is_approval=False, is_confirmation=False):
        """Create Slack message blocks matching HTML preview structure."""
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header, "emoji": False},
            }
        ]

        if not is_approval and not is_confirmation:
            # Initial approval message structure
            blocks.extend(
                [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "_CreditScorer_"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Pipeline:* {pipeline_name}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Model ID:* {run_id}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Stack:* {stack_name}",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"{'✅' if perf_ok else '❌'} *Accuracy:* {accuracy:.3f}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"{'✅' if risk_ok else '❌'} *Risk Score:* {risk_score:.3f}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*F1 Score:* {f1_score:.3f}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*AUC:* {auc_roc:.3f}",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"{'✅ *Bias Check:* Passed' if fairness_ok else f'❌ *Bias Check:* Failed (min DI ratio: {min_di_ratio:.3f})'}",
                        },
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{'MODEL AUTO-APPROVED' if all_ok else 'HUMAN REVIEW REQUIRED'}*",
                        },
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"<!date^{int(time.time())}^{{date_short_pretty}} {{time}}|{datetime.now().isoformat()}>",
                            }
                        ],
                    },
                ]
            )
        elif is_approval:
            # Approval request structure
            blocks.extend(
                [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "_CreditScorer_"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Model ID:* {run_id}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Pipeline:* {pipeline_name}",
                            },
                        ],
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": "_Override deployment? Reply with 'yes' or 'no'_",
                            }
                        ],
                    },
                ]
            )
        else:
            # Confirmation structure
            blocks.extend(
                [
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "_CreditScorer_"},
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"✅ *Model {run_id} has been approved for deployment*",
                        },
                    },
                    {"type": "divider"},
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Model ID:* {run_id}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Pipeline:* {pipeline_name}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Approved by:* {approver}",
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Accuracy:* {accuracy:.3f}",
                            },
                        ],
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Deployment will proceed automatically*",
                        },
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": f"_Model checksum: {run_id}_",
                            }
                        ],
                    },
                ]
            )

        return blocks

    def send_slack_message(message, blocks, ask_question=False):
        """Send Slack message with error handling."""
        if not client.active_stack.alerter:
            return None

        try:
            params = SlackAlerterParameters(
                blocks=blocks,
                payload=SlackAlerterPayload(
                    pipeline_name=pipeline_name,
                    step_name=step_name,
                    stack_name=stack_name,
                ),
            )

            if ask_question:
                print("📱 Asking approval question in Slack...")
                response = client.active_stack.alerter.ask(
                    message, params=params
                )
                print(f"📱 Received Slack response: {response}")
                return response
            else:
                client.active_stack.alerter.post(
                    message=message, params=params
                )
                print("✅ Slack notification sent successfully")
                return True

        except Exception as e:
            print(
                f"⚠️ Slack {'interaction' if ask_question else 'notification'} failed: {e}"
            )
            if "not_in_channel" in str(e):
                print(
                    "💡 Fix: Add your bot to the Slack channel using: /invite @your-bot-name"
                )
            elif "not_allowed_token_type" in str(e):
                print("💡 Fix: Use a Bot User OAuth Token (starts with xoxb-)")
            return None

    # Check for auto-approve from environment variables
    auto_approve = os.environ.get("DEPLOY_APPROVAL", "").lower() == "y"
    env_approver = os.environ.get("APPROVER", "")
    env_rationale = os.environ.get("APPROVAL_RATIONALE", "")

    # Send initial notification
    header = "MODEL AUTO-APPROVED" if all_ok or auto_approve else "HUMAN REVIEW REQUIRED"
    send_slack_message(header, create_blocks("Model Approval"))

    # Determine approval
    if all_ok:
        approved, approver, rationale = (
            True,
            "automated_system",
            "All criteria met",
        )
    elif auto_approve:
        approved, approver, rationale = (
            True,
            env_approver or "automated_ci",
            env_rationale or "Auto-approved via environment variable",
        )
    else:
        response = send_slack_message(
            f"Override deployment for pipeline '{pipeline_name}'?",
            create_blocks("Deployment Approval Required", is_approval=True),
            ask_question=True,
        )

        if response is not None:
            override = (
                response.lower().strip() in ["yes", "y", "true", "1"]
                if isinstance(response, str)
                else bool(response)
            )
            approved, approver = override, "human_via_slack"
            rationale = f"Human {'override' if override else 'rejection'} via Slack for model {run_id}"
            print(
                f"📱 Slack approval result: {'APPROVED' if override else 'REJECTED'}"
            )
        else:
            approved, approver, rationale = (
                False,
                "system",
                "Slack integration failed or no alerter configured",
            )

    # Send confirmation if approved
    if approved and client.active_stack.alerter:
        send_slack_message(
            "✅ Model approved for deployment",
            create_blocks("Deployment Confirmed", is_confirmation=True),
        )
        print("📱 Deployment confirmation sent to Slack")

    if not approved:
        raise RuntimeError(f"🚫 Deployment rejected: {rationale}")

    # Create approval record
    timestamp = datetime.now().isoformat()
    approval_record = {
        "approval_id": f"approval_{timestamp.replace(':', '-')}",
        "timestamp": timestamp,
        "approved": approved,
        "approver": approver,
        "rationale": rationale,
        "model_id": run_id,
        "decision_mode": "automated" if all_ok else "slack_approval",
        "criteria_met": all_ok,
        "failed_criteria": [
            k
            for k, v in {
                "Performance": perf_ok,
                "Fairness": fairness_ok,
                "Risk": risk_ok,
            }.items()
            if not v
        ],
        "bias_detected": bias_flag,
        "key_metrics": {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "auc_roc": auc_roc,
            "normalized_cost": metrics.get("normalized_cost"),
            "risk_score": risk_score,
        },
        "protected_attributes_count": len(fairness_metrics),
        "min_disparate_impact_ratio": min_di_ratio,
        "thresholds_used": approval_thresholds,
        "pipeline_context": {
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "stack_name": stack_name,
            "run_id": run_id,
        },
    }

    print(f"✅ APPROVED by {approver}: {rationale}")
    print(f"📋 Model {run_id} from pipeline: {pipeline_name}")

    return approved, approval_record
