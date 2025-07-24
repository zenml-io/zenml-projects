"""
Output distribution step for delivering summaries and tasks to various platforms.
"""

import os
from typing import List

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from ..utils.models import DeliveryResult, ProcessedData
from ..utils.platforms import (
    DiscordDeliverer,
    LocalDeliverer,
    NotionDeliverer,
    SlackDeliverer,
)

logger = get_logger(__name__)


@step
def output_distribution_step(
    processed_data: ProcessedData,
    output_targets: List[str],
    extract_tasks: bool = True,  # NEW
) -> Annotated[List[DeliveryResult], "delivery_results"]:
    """
    Distribute processed summaries and tasks to specified output targets.

    Args:
        processed_data: Processed summaries and tasks from LangGraph
        output_targets: List of output destinations (notion, slack, github)
        extract_tasks: Whether to include tasks in delivery

    Returns:
        List[DeliveryResult]: Results from each delivery attempt
    """

    logger.info(f"Starting output distribution to targets: {output_targets}")

    # Decide once whether tasks should be delivered
    should_deliver_tasks = extract_tasks and bool(processed_data.tasks)  # NEW

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
            if should_deliver_tasks:  # NEW
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
        if should_deliver_tasks:  # NEW
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
                # ONE consolidated message for summaries (+ tasks if enabled)
                res = await discord_deliverer.deliver_consolidated(
                    [s.model_dump() for s in processed_data.summaries],
                    [t.model_dump() for t in processed_data.tasks]
                    if should_deliver_tasks
                    else None,
                )
                # Allow discord.py to finish closing its aiohttp connector
                await asyncio.sleep(0.1)
                return [res]

            # Execute the batch and extend overall delivery results
            batch_results = asyncio.run(_discord_batch())
            delivery_results.extend(batch_results)
        else:
            logger.warning(
                "DISCORD_BOT_TOKEN not found, skipping Discord delivery"
            )

    # Local markdown delivery
    if "local" in output_targets:
        export_dir = os.getenv("LOCAL_OUTPUT_DIR", "discord_summaries")
        local_deliverer = LocalDeliverer(export_dir)

        # Consolidated delivery (summaries + optional tasks)
        result = local_deliverer.deliver_consolidated(
            [s.model_dump() for s in processed_data.summaries],
            [t.model_dump() for t in processed_data.tasks]
            if should_deliver_tasks
            else None,
            filename_prefix="Daily_Team_Digest",
        )
        delivery_results.append(result)

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
