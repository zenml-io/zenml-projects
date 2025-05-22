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
    # Get pipeline context information
    step_context = get_step_context()
    pipeline_name = step_context.pipeline.name
    step_name = step_context.step_run.name
    stack_name = Client().active_stack.name
    run_id = str(step_context.pipeline_run.id)

    # Extract key metrics
    metrics = evaluation_results.get("metrics", {})
    fairness_data = evaluation_results.get("fairness", {})
    fairness_metrics = fairness_data.get("fairness_metrics", {})
    bias_flag = fairness_data.get("bias_flag", False)

    accuracy = metrics.get("accuracy", 0)
    f1_score = metrics.get("f1_score", 0)
    auc_roc = metrics.get("auc_roc", 0)
    risk_score = risk_scores.get("overall", 1)
    max_disparity = (
        max(
            [
                abs(attr.get("selection_rate_disparity", 0))
                for attr in fairness_metrics.values()
            ]
        )
        if fairness_metrics
        else 0
    )

    # Generate model checksum for identification
    model_id = run_id  # Use first 8 chars of run ID as model ID

    # Approval criteria checks
    perf_ok = accuracy >= approval_thresholds.get("accuracy", 0.7)
    fairness_ok = not bias_flag and max_disparity <= approval_thresholds.get(
        "max_disparity", 0.2
    )
    risk_ok = risk_score <= approval_thresholds.get("risk_score", 0.4)
    all_ok = perf_ok and fairness_ok and risk_ok

    # Create Slack message with enhanced pipeline context
    header_text = (
        "*MODEL AUTO-APPROVED*" if all_ok else "*HUMAN REVIEW REQUIRED*"
    )

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "Model Approval",
                "emoji": False,
            },
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": header_text}},
        {"type": "divider"},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Pipeline:* {pipeline_name}"},
                {"type": "mrkdwn", "text": f"*Model ID:* {model_id}"},
                {"type": "mrkdwn", "text": f"*Stack:* {stack_name}"},
                {"type": "mrkdwn", "text": f"*Run ID:* {run_id}"},
            ],
        },
        {"type": "divider"},
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
                    "text": f"{'✅' if fairness_ok else '❌'} *F1 Score:* {f1_score:.3f}",
                },
                {
                    "type": "mrkdwn",
                    "text": f"*AUC:* {auc_roc:.3f}",
                },
            ],
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Bias Check:* {'✅ No bias detected' if fairness_ok else f'❌ Max disparity: {max_disparity:.3f}'}",
                },
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Pipeline: {pipeline_name} • <!date^{int(time.time())}^{{date_short_pretty}} {{time}}|{datetime.now().isoformat()}>",
                },
            ],
        },
    ]

    # Include pipeline info in payload
    params = SlackAlerterParameters(
        blocks=blocks,
        payload=SlackAlerterPayload(
            pipeline_name=pipeline_name,
            step_name=step_name,
            stack_name=stack_name,
        ),
    )

    alerter = Client().active_stack.alerter

    if alerter:
        try:
            alerter.post(message=header_text, params=params)
            print("✅ Slack notification sent successfully")
        except Exception as e:
            print(f"⚠️  Slack notification failed: {e}")
            print("Continuing without Slack notification...")

    if all_ok:
        approved, approver, rationale = (
            True,
            "automated_system",
            "All criteria met",
        )
    else:
        if alerter:
            try:
                # Simplified approval question with clearer formatting
                approval_blocks = [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "*Deployment Approval Required*",
                        },
                    },
                    {"type": "divider"},
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Model ID:* {model_id}",
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

                approval_params = SlackAlerterParameters(
                    blocks=approval_blocks
                )

                # Use a more direct question for the alerter.ask function
                question = (
                    f"Override deployment for pipeline '{pipeline_name}'?"
                )
                print("📱 Asking approval question in Slack...")
                response = alerter.ask(question, params=approval_params)
                print(f"📱 Received Slack response: {response}")

                # Handle various response formats
                if isinstance(response, str):
                    response_lower = response.lower().strip()
                    override = response_lower in ["yes", "y", "true", "1"]
                else:
                    override = bool(response)

                approved = override
                approver = "human_via_slack"
                rationale = (
                    f"Human override via Slack for model {model_id}"
                    if override
                    else f"Rejected via Slack for model {model_id}"
                )
                print(
                    f"📱 Slack approval result: {'APPROVED' if override else 'REJECTED'}"
                )

            except Exception as e:
                print(f"❌ Slack interaction failed: {e}")
                if "not_in_channel" in str(e):
                    print(
                        "💡 Fix: Add your bot to the Slack channel using: /invite @your-bot-name"
                    )
                elif "not_allowed_token_type" in str(e):
                    print(
                        "💡 Fix: Use a Bot User OAuth Token (starts with xoxb-)"
                    )
                print("❌ Cannot get human approval - deployment blocked")
                approved, approver, rationale = (
                    False,
                    "system",
                    f"Slack integration failed: {str(e)}",
                )
        else:
            approved, approver, rationale = (
                False,
                "system",
                "No alerter configured - blocked",
            )

    # Send confirmation message if approved
    if approved and alerter:
        try:
            confirmation_blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Deployment Confirmed",
                        "emoji": False,
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "_CreditScorer_",
                        }
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"✅ *Model {model_id} has been approved for deployment*",
                    },
                },
                {"type": "divider"},
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Model ID:* {model_id}"},
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
                        "text": f"*Deployment will proceed automatically*\n_Model checksum: {run_id}_",
                    },
                },
            ]

            confirmation_params = SlackAlerterParameters(
                blocks=confirmation_blocks
            )
            alerter.post(
                "✅ Model approved for deployment",
                params=confirmation_params,
            )
            print("📱 Deployment confirmation sent to Slack")

        except Exception as e:
            print(f"⚠️  Could not send confirmation message: {e}")

    if not approved:
        raise RuntimeError(f"🚫 Deployment rejected: {rationale}")

    # Complete approval record with pipeline context
    timestamp = datetime.now().isoformat()
    approval_record = {
        "approval_id": f"approval_{timestamp.replace(':', '-')}",
        "timestamp": timestamp,
        "approved": approved,
        "approver": approver,
        "rationale": rationale,
        "model_id": model_id,
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
        "max_bias_disparity": max_disparity,
        "thresholds_used": approval_thresholds,
        # Pipeline traceability
        "pipeline_context": {
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "stack_name": stack_name,
            "run_id": run_id,
        },
    }

    print(f"✅ APPROVED by {approver}: {rationale}")
    print(f"📋 Model {model_id} from pipeline: {pipeline_name}")

    return approved, approval_record
