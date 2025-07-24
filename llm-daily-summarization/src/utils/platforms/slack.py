"""
Slack platform deliverer (summary & tasks).
"""

from __future__ import annotations

from typing import Any, Dict, List

from zenml.logger import get_logger

from ..models import DeliveryResult

logger = get_logger(__name__)


class SlackDeliverer:
    """Delivers content to Slack via webhook (mocked by default)."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    # ------------------------------------------------------------------ #
    def deliver_summary(self, summary_data: Dict[str, Any]) -> DeliveryResult:
        """Send summary to Slack (mocked)."""
        try:
            logger.info(
                "Delivering summary to Slack: %s", summary_data["title"]
            )
            message = self._format_summary_for_slack(summary_data)
            logger.info("Slack message content: %s...", message[:100])

            return DeliveryResult(
                target="slack",
                success=True,
                delivered_items=[summary_data["title"]],
                failed_items=[],
                delivery_url="https://slack.com/mock-channel",
                error_message=None,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to deliver to Slack: %s", exc)
            return DeliveryResult(
                target="slack",
                success=False,
                delivered_items=[],
                failed_items=[summary_data["title"]],
                delivery_url=None,
                error_message=str(exc),
            )

    def deliver_tasks(self, tasks: List[Dict[str, Any]]) -> DeliveryResult:
        """Send tasks list to Slack (mocked)."""
        try:
            logger.info("Delivering %d tasks to Slack", len(tasks))
            message = self._format_tasks_for_slack(tasks)
            logger.info("Slack tasks message: %s...", message[:100])

            return DeliveryResult(
                target="slack",
                success=True,
                delivered_items=[t["title"] for t in tasks],
                failed_items=[],
                delivery_url="https://slack.com/mock-channel",
                error_message=None,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to deliver tasks to Slack: %s", exc)
            return DeliveryResult(
                target="slack",
                success=False,
                delivered_items=[],
                failed_items=[t["title"] for t in tasks],
                delivery_url=None,
                error_message=str(exc),
            )

    # ------------------------------------------------------------------ #
    # Helper formatting functions (copied unchanged)                     #
    # ------------------------------------------------------------------ #
    def _format_summary_for_slack(self, summary_data: Dict[str, Any]) -> str:
        message = (
            f"📝 *{summary_data['title']}*\n\n{summary_data['content']}\n\n"
        )
        if summary_data.get("key_points"):
            message += "*Key Points:*\n"
            for point in summary_data["key_points"]:
                message += f"• {point}\n"
            message += "\n"
        if summary_data.get("participants"):
            message += (
                f"*Participants:* {', '.join(summary_data['participants'])}\n"
            )
        return message

    def _format_tasks_for_slack(self, tasks: List[Dict[str, Any]]) -> str:
        if not tasks:
            return "No action items identified today."

        message = f"✅ *Action Items ({len(tasks)} tasks)*\n\n"
        for task in tasks:
            emoji = {"high": "🔥", "medium": "⚡", "low": "📝"}.get(
                task.get("priority", "medium"), "📝"
            )
            message += f"{emoji} *{task['title']}*\n   {task['description']}\n"
            if task.get("assignee"):
                message += f"   👤 Assigned to: {task['assignee']}\n"
            if task.get("due_date"):
                message += f"   📅 Due: {task['due_date']}\n"
            message += "\n"
        return message
