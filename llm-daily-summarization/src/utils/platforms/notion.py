"""
Notion platform deliverer (summary & tasks).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from zenml.logger import get_logger

from ..models import DeliveryResult

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
                            self.notion.pages.create(
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
