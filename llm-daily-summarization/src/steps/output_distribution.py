"""
Output distribution step for delivering summaries and tasks to various platforms.
"""

import os
from typing import Any, Dict, List

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from ..utils.models import DeliveryResult, ProcessedData

logger = get_logger(__name__)


class NotionDeliverer:
    """Delivers content to Notion."""

    def __init__(self, api_token: str):  # noqa: D107
        self.api_token = api_token
        # Import here to avoid import errors if notion-client is not installed
        try:
            from notion_client import Client

            self.notion = Client(auth=api_token)
        except ImportError:
            logger.warning(
                "notion-client not installed, using mock implementation"
            )
            self.notion = None

    def deliver_summary(self, summary_data: Dict[str, Any]) -> DeliveryResult:
        """Deliver summary to Notion as a page."""
        try:
            logger.info(
                f"Delivering summary to Notion: {summary_data['title']}"
            )

            if self.notion is None:
                logger.warning(
                    "Using mock implementation - notion-client not available"
                )
                return self._mock_summary_delivery(summary_data)

            # Get or create database for summaries
            summaries_db_id = os.getenv("NOTION_SUMMARIES_DB_ID")
            if not summaries_db_id:
                logger.warning(
                    "NOTION_SUMMARIES_DB_ID not set, using mock implementation"
                )
                return self._mock_summary_delivery(summary_data)

            # Create summary page in Notion database with basic properties
            # Only use Title (which should exist in most databases)
            page_properties = {
                "Title": {
                    "title": [{"text": {"content": summary_data["title"]}}]
                },
            }

            # Try to add optional properties if they exist
            try:
                # Try to detect and use existing properties by checking database schema
                # For now, we'll just use the basic title and put everything in content
                pass
            except Exception:
                # If property detection fails, just use title
                pass

            # Create comprehensive page content
            content_blocks = [
                # Summary content
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": summary_data["content"]},
                            }
                        ]
                    },
                },
                # Metadata section
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": "📊 Summary Details"},
                            }
                        ]
                    },
                },
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": f"• Word Count: {summary_data['word_count']}\n"
                                },
                            },
                            {
                                "type": "text",
                                "text": {
                                    "content": f"• Confidence: {summary_data['confidence_score']:.1%}\n"
                                },
                            },
                            {
                                "type": "text",
                                "text": {
                                    "content": f"• Participants: {', '.join(summary_data.get('participants', []))}\n"
                                },
                            },
                            {
                                "type": "text",
                                "text": {
                                    "content": f"• Topics: {', '.join(summary_data.get('topics', []))}"
                                },
                            },
                        ]
                    },
                },
            ]

            # Add key points as bulleted list
            if summary_data.get("key_points"):
                content_blocks.append(
                    {
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": "🔍 Key Points"},
                                }
                            ]
                        },
                    }
                )

                for point in summary_data["key_points"]:
                    content_blocks.append(
                        {
                            "object": "block",
                            "type": "bulleted_list_item",
                            "bulleted_list_item": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {"content": point},
                                    }
                                ]
                            },
                        }
                    )

            # Create the page
            page = self.notion.pages.create(
                parent={"database_id": summaries_db_id},
                properties=page_properties,
                children=content_blocks,
            )

            page_url = page.get("url", "https://notion.so")
            logger.info(f"Successfully created summary page: {page_url}")

            return DeliveryResult(
                target="notion",
                success=True,
                delivered_items=[summary_data["title"]],
                failed_items=[],
                delivery_url=page_url,
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Failed to deliver summary to Notion: {e}")
            return DeliveryResult(
                target="notion",
                success=False,
                delivered_items=[],
                failed_items=[summary_data["title"]],
                delivery_url=None,
                error_message=str(e),
            )

    def _mock_summary_delivery(
        self, summary_data: Dict[str, Any]
    ) -> DeliveryResult:
        """Mock implementation for summary delivery."""
        return DeliveryResult(
            target="notion",
            success=True,
            delivered_items=[summary_data["title"]],
            failed_items=[],
            delivery_url="https://notion.so/mock-page",
            error_message=None,
        )

    def deliver_tasks(self, tasks: List[Dict[str, Any]]) -> DeliveryResult:
        """Deliver tasks to Notion as database entries."""
        try:
            logger.info(f"Delivering {len(tasks)} tasks to Notion")

            if self.notion is None:
                logger.warning(
                    "Using mock implementation - notion-client not available"
                )
                return self._mock_tasks_delivery(tasks)

            # Get database for tasks
            tasks_db_id = os.getenv("NOTION_TASKS_DB_ID")
            if not tasks_db_id:
                logger.warning(
                    "NOTION_TASKS_DB_ID not set, using mock implementation"
                )
                return self._mock_tasks_delivery(tasks)

            delivered_items = []
            failed_items = []

            # Create each task as a database entry
            for task in tasks:
                try:
                    # Map priority to Notion select options
                    priority_mapping = {
                        "high": "High",
                        "medium": "Medium",
                        "low": "Low",
                    }
                    priority_notion = priority_mapping.get(
                        task.get("priority", "medium"), "Medium"
                    )

                    # Try different title property names that Notion databases commonly use
                    title_content = {
                        "title": [{"text": {"content": task["title"]}}]
                    }
                    common_title_props = [
                        "Name",
                        "Title",
                        "Task",
                        "name",
                        "title",
                    ]

                    # Start with the most common one
                    current_title_prop = "Name"
                    task_properties = {current_title_prop: title_content}

                    # Create task content with all details
                    task_content = [
                        # Task description
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": task["description"]
                                        },
                                    }
                                ]
                            },
                        },
                        # Task details
                        {
                            "object": "block",
                            "type": "heading_3",
                            "heading_3": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {"content": "📋 Task Details"},
                                    }
                                ]
                            },
                        },
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": f"• Priority: {task.get('priority', 'medium').upper()}\n"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": f"• Confidence: {task['confidence_score']:.1%}\n"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": f"• Assignee: {task.get('assignee', 'Unassigned')}\n"
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": {
                                            "content": f"• Due Date: {task.get('due_date', 'Not specified')}"
                                        },
                                    },
                                ]
                            },
                        },
                    ]

                    # Try to create the task page with different title properties
                    page_created = False
                    for title_prop_attempt in common_title_props:
                        try:
                            current_properties = {
                                title_prop_attempt: title_content
                            }
                            page = self.notion.pages.create(
                                parent={"database_id": tasks_db_id},
                                properties=current_properties,
                                children=task_content,
                            )
                            delivered_items.append(task["title"])
                            logger.info(
                                f"Successfully created task '{task['title']}' using property '{title_prop_attempt}'"
                            )
                            page_created = True
                            break
                        except Exception as prop_error:
                            logger.debug(
                                f"Failed to create task with property '{title_prop_attempt}': {prop_error}"
                            )
                            continue

                    if not page_created:
                        raise Exception(
                            f"Failed to create task with any title property. Tried: {common_title_props}"
                        )

                except Exception as task_error:
                    logger.error(
                        f"Failed to create task '{task['title']}': {task_error}"
                    )
                    failed_items.append(task["title"])

            success = len(delivered_items) > 0
            db_url = f"https://notion.so/{tasks_db_id.replace('-', '')}"

            return DeliveryResult(
                target="notion",
                success=success,
                delivered_items=delivered_items,
                failed_items=failed_items,
                delivery_url=db_url if success else None,
                error_message=None
                if success
                else "Failed to create any tasks",
            )

        except Exception as e:
            logger.error(f"Failed to deliver tasks to Notion: {e}")
            return DeliveryResult(
                target="notion",
                success=False,
                delivered_items=[],
                failed_items=[task["title"] for task in tasks],
                delivery_url=None,
                error_message=str(e),
            )

    def _mock_tasks_delivery(
        self, tasks: List[Dict[str, Any]]
    ) -> DeliveryResult:
        """Mock implementation for task delivery."""
        delivered_items = [task["title"] for task in tasks]
        return DeliveryResult(
            target="notion",
            success=True,
            delivered_items=delivered_items,
            failed_items=[],
            delivery_url="https://notion.so/mock-tasks-database",
            error_message=None,
        )


class SlackDeliverer:
    """Delivers content to Slack."""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url

    def deliver_summary(self, summary_data: Dict[str, Any]) -> DeliveryResult:
        """Deliver summary to Slack."""

        try:
            logger.info(
                f"Delivering summary to Slack: {summary_data['title']}"
            )

            # Format summary for Slack
            message = self._format_summary_for_slack(summary_data)

            # Mock delivery
            logger.info(f"Slack message content: {message[:100]}...")

            return DeliveryResult(
                target="slack",
                success=True,
                delivered_items=[summary_data["title"]],
                failed_items=[],
                delivery_url="https://slack.com/mock-channel",
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Failed to deliver to Slack: {e}")
            return DeliveryResult(
                target="slack",
                success=False,
                delivered_items=[],
                failed_items=[summary_data["title"]],
                delivery_url=None,
                error_message=str(e),
            )

    def _format_summary_for_slack(self, summary_data: Dict[str, Any]) -> str:
        """Format summary for Slack message."""

        message = f"📝 *{summary_data['title']}*\n\n"
        message += f"{summary_data['content']}\n\n"

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

    def deliver_tasks(self, tasks: List[Dict[str, Any]]) -> DeliveryResult:
        """Deliver tasks to Slack."""

        try:
            logger.info(f"Delivering {len(tasks)} tasks to Slack")

            # Format tasks for Slack
            message = self._format_tasks_for_slack(tasks)

            # Mock delivery
            logger.info(f"Slack tasks message: {message[:100]}...")

            return DeliveryResult(
                target="slack",
                success=True,
                delivered_items=[task["title"] for task in tasks],
                failed_items=[],
                delivery_url="https://slack.com/mock-channel",
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Failed to deliver tasks to Slack: {e}")
            return DeliveryResult(
                target="slack",
                success=False,
                delivered_items=[],
                failed_items=[task["title"] for task in tasks],
                delivery_url=None,
                error_message=str(e),
            )

    def _format_tasks_for_slack(self, tasks: List[Dict[str, Any]]) -> str:
        """Format tasks for Slack message."""

        if not tasks:
            return "No action items identified today."

        message = f"✅ *Action Items ({len(tasks)} tasks)*\n\n"

        for i, task in enumerate(tasks, 1):
            priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📝"}.get(
                task.get("priority", "medium"), "📝"
            )
            message += f"{priority_emoji} *{task['title']}*\n"
            message += f"   {task['description']}\n"

            if task.get("assignee"):
                message += f"   👤 Assigned to: {task['assignee']}\n"

            if task.get("due_date"):
                message += f"   📅 Due: {task['due_date']}\n"

            message += "\n"

        return message


class DiscordDeliverer:
    """Delivers content to Discord channels."""

    def __init__(
        self,
        token: str,
        channel_id: str,
        summary_config: Dict[str, Any] = None,
    ):
        """Initialize Discord deliverer.

        Args:
            token: Discord bot token
            channel_id: Discord channel ID for posting summaries
            summary_config: Configuration for summary formatting
        """
        self.token = token
        self.channel_id = channel_id
        self.summary_config = summary_config or {}

        # Import here to avoid import errors if discord.py is not installed
        try:
            from ..utils.chat_clients import DiscordClient

            self.discord_client = DiscordClient(token)
        except ImportError:
            logger.warning(
                "discord.py not installed, using mock implementation"
            )
            self.discord_client = None

    async def deliver_summary(
        self, summary_data: Dict[str, Any]
    ) -> DeliveryResult:
        """Deliver summary to Discord channel."""
        try:
            logger.info(
                f"Delivering summary to Discord channel {self.channel_id}: {summary_data['title']}"
            )

            if self.discord_client is None:
                logger.warning(
                    "Using mock implementation - discord.py not available"
                )
                return self._mock_summary_delivery(summary_data)

            # Format summary for Discord
            discord_message = self._format_summary_for_discord(summary_data)

            # Apply any length constraints from config
            max_length = self.summary_config.get("max_length", 1900)
            if (
                self.summary_config.get("truncate_to_limit", False)
                and len(discord_message) > max_length
            ):
                discord_message = discord_message[: max_length - 3] + "..."

            # Post to Discord
            success = await self.discord_client.post_summary(
                self.channel_id, discord_message, max_length=max_length
            )

            if success:
                return DeliveryResult(
                    target="discord",
                    success=True,
                    delivered_items=[summary_data["title"]],
                    failed_items=[],
                    delivery_url=f"https://discord.com/channels/@me/{self.channel_id}",
                    error_message=None,
                )
            else:
                return DeliveryResult(
                    target="discord",
                    success=False,
                    delivered_items=[],
                    failed_items=[summary_data["title"]],
                    delivery_url=None,
                    error_message="Failed to post to Discord channel",
                )

        except Exception as e:
            logger.error(f"Failed to deliver summary to Discord: {e}")
            return DeliveryResult(
                target="discord",
                success=False,
                delivered_items=[],
                failed_items=[summary_data["title"]],
                delivery_url=None,
                error_message=str(e),
            )

    def _format_summary_for_discord(self, summary_data: Dict[str, Any]) -> str:
        """Format summary for Discord message based on configuration."""
        # Check if we should use a compact format
        if self.summary_config.get("format", "full") == "compact":
            # Compact format - just key points
            message = f"📝 **{summary_data['title']}**\n\n"

            if summary_data.get("key_points"):
                message += "**Key Points:**\n"
                max_points = self.summary_config.get("max_key_points", 5)
                for point in summary_data["key_points"][:max_points]:
                    message += f"• {point}\n"

            return message

        # Full format (default)
        message = f"📝 **{summary_data['title']}**\n\n"

        # Include summary content if configured
        if self.summary_config.get("include_content", True):
            content = summary_data["content"]
            max_content_length = self.summary_config.get(
                "max_content_length", 800
            )
            if len(content) > max_content_length:
                content = content[: max_content_length - 3] + "..."
            message += f"{content}\n\n"

        # Add key points if available
        if summary_data.get("key_points") and self.summary_config.get(
            "include_key_points", True
        ):
            message += "**Key Points:**\n"
            max_points = self.summary_config.get("max_key_points", 5)
            for point in summary_data["key_points"][:max_points]:
                message += f"• {point}\n"
            message += "\n"

        # Add metadata based on config
        if self.summary_config.get("include_metadata", True):
            metadata_items = []

            if summary_data.get("participants") and self.summary_config.get(
                "show_participants", True
            ):
                participants = ", ".join(summary_data["participants"][:5])
                if len(summary_data["participants"]) > 5:
                    participants += (
                        f" (+{len(summary_data['participants']) - 5} more)"
                    )
                metadata_items.append(f"👥 **Participants:** {participants}")

            if summary_data.get("topics") and self.summary_config.get(
                "show_topics", True
            ):
                topics = ", ".join(summary_data["topics"][:3])
                metadata_items.append(f"🏷️ **Topics:** {topics}")

            if summary_data.get("word_count") and self.summary_config.get(
                "show_stats", False
            ):
                metadata_items.append(
                    f"📊 **Words:** {summary_data['word_count']}"
                )

            if metadata_items:
                message += "\n".join(metadata_items) + "\n"

        return message.strip()

    def _mock_summary_delivery(
        self, summary_data: Dict[str, Any]
    ) -> DeliveryResult:
        """Mock implementation for summary delivery."""
        return DeliveryResult(
            target="discord",
            success=True,
            delivered_items=[summary_data["title"]],
            failed_items=[],
            delivery_url=f"https://discord.com/channels/@me/{self.channel_id}",
            error_message=None,
        )

    async def deliver_tasks(
        self, tasks: List[Dict[str, Any]]
    ) -> DeliveryResult:
        """Deliver tasks to Discord channel."""
        try:
            logger.info(
                f"Delivering {len(tasks)} tasks to Discord channel {self.channel_id}"
            )

            if self.discord_client is None:
                logger.warning(
                    "Using mock implementation - discord.py not available"
                )
                return self._mock_tasks_delivery(tasks)

            # Format tasks for Discord
            discord_message = self._format_tasks_for_discord(tasks)

            # Apply any length constraints from config
            max_length = self.summary_config.get("max_length", 1900)

            # Post to Discord
            success = await self.discord_client.post_summary(
                self.channel_id, discord_message, max_length=max_length
            )

            if success:
                return DeliveryResult(
                    target="discord",
                    success=True,
                    delivered_items=[task["title"] for task in tasks],
                    failed_items=[],
                    delivery_url=f"https://discord.com/channels/@me/{self.channel_id}",
                    error_message=None,
                )
            else:
                return DeliveryResult(
                    target="discord",
                    success=False,
                    delivered_items=[],
                    failed_items=[task["title"] for task in tasks],
                    delivery_url=None,
                    error_message="Failed to post tasks to Discord channel",
                )

        except Exception as e:
            logger.error(f"Failed to deliver tasks to Discord: {e}")
            return DeliveryResult(
                target="discord",
                success=False,
                delivered_items=[],
                failed_items=[task["title"] for task in tasks],
                delivery_url=None,
                error_message=str(e),
            )

    def _format_tasks_for_discord(self, tasks: List[Dict[str, Any]]) -> str:
        """Format tasks for Discord message."""
        if not tasks:
            return "No action items identified today."

        message = f"✅ **Action Items ({len(tasks)} tasks)**\n\n"

        # Limit number of tasks based on config
        max_tasks = self.summary_config.get("max_tasks_display", 10)
        displayed_tasks = tasks[:max_tasks]

        for i, task in enumerate(displayed_tasks, 1):
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                task.get("priority", "medium"), "⚪"
            )

            message += f"{i}. {priority_emoji} **{task['title']}**\n"

            if self.summary_config.get("include_task_details", True):
                # Truncate description if needed
                description = task["description"]
                max_desc_length = self.summary_config.get(
                    "max_task_description_length", 100
                )
                if len(description) > max_desc_length:
                    description = description[: max_desc_length - 3] + "..."
                message += f"   {description}\n"

                if task.get("assignee") and self.summary_config.get(
                    "show_assignees", True
                ):
                    message += f"   👤 Assigned to: {task['assignee']}\n"

                if task.get("due_date") and self.summary_config.get(
                    "show_due_dates", False
                ):
                    message += f"   📅 Due: {task['due_date']}\n"

            message += "\n"

        if len(tasks) > max_tasks:
            message += f"... and {len(tasks) - max_tasks} more tasks\n"

        return message.strip()

    def _mock_tasks_delivery(
        self, tasks: List[Dict[str, Any]]
    ) -> DeliveryResult:
        """Mock implementation for task delivery."""
        return DeliveryResult(
            target="discord",
            success=True,
            delivered_items=[task["title"] for task in tasks],
            failed_items=[],
            delivery_url=f"https://discord.com/channels/@me/{self.channel_id}",
            error_message=None,
        )


@step
def output_distribution_step(
    processed_data: ProcessedData, output_targets: List[str]
) -> Annotated[List[DeliveryResult], "delivery_results"]:
    """
    Distribute processed summaries and tasks to specified output targets.

    Args:
        processed_data: Processed summaries and tasks from LangGraph
        output_targets: List of output destinations (notion, slack, github)

    Returns:
        List[DeliveryResult]: Results from each delivery attempt
    """

    logger.info(f"Starting output distribution to targets: {output_targets}")

    delivery_results = []

    # Notion delivery
    if "notion" in output_targets:
        notion_token = os.getenv("NOTION_TOKEN")
        if notion_token:
            notion_deliverer = NotionDeliverer(notion_token)

            # Deliver summaries
            for summary in processed_data.summaries:
                result = notion_deliverer.deliver_summary(summary.model_dump())
                delivery_results.append(result)

            # Deliver tasks
            if processed_data.tasks:
                task_result = notion_deliverer.deliver_tasks(
                    [task.model_dump() for task in processed_data.tasks]
                )
                delivery_results.append(task_result)
        else:
            logger.warning("NOTION_TOKEN not found, skipping Notion delivery")

    # Slack delivery
    if "slack" in output_targets:
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        slack_deliverer = SlackDeliverer(slack_webhook)

        # Deliver summaries
        for summary in processed_data.summaries:
            result = slack_deliverer.deliver_summary(summary.model_dump())
            delivery_results.append(result)

        # Deliver tasks
        if processed_data.tasks:
            task_result = slack_deliverer.deliver_tasks(
                [task.model_dump() for task in processed_data.tasks]
            )
            delivery_results.append(task_result)

    # Discord delivery
    if "discord" in output_targets:
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        discord_channel_id = os.getenv(
            "DISCORD_SUMMARY_CHANNEL_ID", "908270509127499776"
        )

        if discord_token:
            # Load Discord summary configuration
            discord_summary_config = {
                "format": os.getenv(
                    "DISCORD_SUMMARY_FORMAT", "full"
                ),  # full or compact
                "max_length": int(
                    os.getenv("DISCORD_MAX_MESSAGE_LENGTH", "1900")
                ),
                "include_content": os.getenv(
                    "DISCORD_INCLUDE_CONTENT", "true"
                ).lower()
                == "true",
                "include_key_points": os.getenv(
                    "DISCORD_INCLUDE_KEY_POINTS", "true"
                ).lower()
                == "true",
                "include_metadata": os.getenv(
                    "DISCORD_INCLUDE_METADATA", "true"
                ).lower()
                == "true",
                "max_key_points": int(
                    os.getenv("DISCORD_MAX_KEY_POINTS", "5")
                ),
                "max_content_length": int(
                    os.getenv("DISCORD_MAX_CONTENT_LENGTH", "800")
                ),
                "show_participants": os.getenv(
                    "DISCORD_SHOW_PARTICIPANTS", "true"
                ).lower()
                == "true",
                "show_topics": os.getenv("DISCORD_SHOW_TOPICS", "true").lower()
                == "true",
                "show_stats": os.getenv("DISCORD_SHOW_STATS", "false").lower()
                == "true",
                "include_task_details": os.getenv(
                    "DISCORD_INCLUDE_TASK_DETAILS", "true"
                ).lower()
                == "true",
                "max_tasks_display": int(
                    os.getenv("DISCORD_MAX_TASKS_DISPLAY", "10")
                ),
                "max_task_description_length": int(
                    os.getenv("DISCORD_MAX_TASK_DESC_LENGTH", "100")
                ),
                "show_assignees": os.getenv(
                    "DISCORD_SHOW_ASSIGNEES", "true"
                ).lower()
                == "true",
                "show_due_dates": os.getenv(
                    "DISCORD_SHOW_DUE_DATES", "false"
                ).lower()
                == "true",
            }

            discord_deliverer = DiscordDeliverer(
                discord_token, discord_channel_id, discord_summary_config
            )

            # Handle async Discord delivery
            import asyncio

            async def _discord_batch():
                """Run all Discord deliveries in one event loop to avoid
                creating multiple connectors which can leak resources."""
                results: List[DeliveryResult] = []

                # Deliver summaries
                for summary in processed_data.summaries:
                    res = await discord_deliverer.deliver_summary(
                        summary.model_dump()
                    )
                    results.append(res)

                # Deliver tasks
                if processed_data.tasks:
                    res = await discord_deliverer.deliver_tasks(
                        [task.model_dump() for task in processed_data.tasks]
                    )
                    results.append(res)

                return results

            # Execute the batch and extend overall delivery results
            batch_results = asyncio.run(_discord_batch())
            delivery_results.extend(batch_results)
        else:
            logger.warning(
                "DISCORD_BOT_TOKEN not found, skipping Discord delivery"
            )

    # GitHub delivery
    if "github" in output_targets:
        logger.info("GitHub delivery not implemented yet")
        # Would implement GitHub Issues API delivery

    successful_deliveries = sum(
        1 for result in delivery_results if result.success
    )
    logger.info(
        f"Output distribution complete: {successful_deliveries}/{len(delivery_results)} successful"
    )

    return delivery_results
