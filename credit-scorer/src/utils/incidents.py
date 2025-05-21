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
from typing import Any, Dict, Optional

from src.constants import (
    MODEL_NAME,
    SLACK_BOT_TOKEN,
    SLACK_CHANNEL,
)
from src.utils.storage import save_artifact_to_modal


def create_incident_report(
    incident_data: Dict[str, Any],
    model_version: Optional[str] = None,
    incident_log_path: str = "docs/risk/incident_log.json",
) -> Dict[str, Any]:
    """Create an incident report and save it to the compliance system.

    Can be used from both the deployed Modal service and internal pipeline code.

    Args:
        incident_data: Details about the incident
        model_version: Version of the model
        incident_log_path: Path to the incident log file

    Returns:
        Status information about the incident report
    """
    # Format incident report
    incident = {
        "incident_id": f"incident_{datetime.now().isoformat().replace(':', '-')}",
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "severity": incident_data.get("severity", "medium"),
        "description": incident_data.get(
            "description", "Unspecified incident"
        ),
        "source": incident_data.get("source", "unknown"),
        "data": incident_data,
    }

    try:
        # 1. Save to local compliance directory first (for non-Modal contexts)
        existing_incidents = []
        if Path(incident_log_path).exists():
            try:
                with open(incident_log_path, "r") as f:
                    existing_incidents = json.load(f)
            except json.JSONDecodeError:
                existing_incidents = []

        existing_incidents.append(incident)

        with open(incident_log_path, "w") as f:
            json.dump(existing_incidents, f, indent=2)

        # 3. Try to save to Modal volume if available
        try:
            save_artifact_to_modal(
                artifact=existing_incidents,
                artifact_path=incident_log_path,
            )
            persisted_to_modal = True
        except Exception:
            # May not be running in a Modal context
            persisted_to_modal = False

        if SLACK_BOT_TOKEN and incident_data.get("severity", "") in [
            "high",
            "critical",
        ]:
            try:
                from slack_sdk import WebClient

                slack_client = WebClient(token=SLACK_BOT_TOKEN)

                # Create a well-formatted Slack message
                emoji = {"high": "🔴", "critical": "🚨"}.get(
                    incident_data.get("severity"), "⚠️"
                )

                message = (
                    f"{emoji} *Incident Report: {incident['description']}*\n"
                    f">*Severity:* {incident['severity']}\n"
                    f">*Source:* {incident['source']}\n"
                    f">*Model Version:* {incident['model_version']}\n"
                    f">*Time:* {incident['timestamp']}\n"
                    f">*Incident ID:* {incident['incident_id']}"
                )

                if "details" in incident_data:
                    message += f"\n>*Details:* {incident_data['details']}"

                slack_client.chat_postMessage(
                    channel=SLACK_CHANNEL,
                    text=message,
                )

                # Add slack notification status to result
                incident["slack_notified"] = True

            except Exception as e:
                print(f"Failed to send Slack notification: {e}")
                incident["slack_notified"] = False

        return {
            "status": "reported",
            "incident_id": incident["incident_id"],
            "persisted": True,
            "persisted_to_modal": persisted_to_modal,
        }
    except Exception as e:
        print(f"Error reporting incident: {e}")
        return {"status": "error", "message": str(e)}
