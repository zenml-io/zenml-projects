"""
Data ingestion step for chat platforms (Discord, Slack).
"""

import asyncio
import os
from datetime import UTC, datetime
from typing import Dict, List, Optional

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from ..utils.chat_clients import DiscordClient, SlackClient
from ..utils.models import RawConversationData

logger = get_logger(__name__)


@step
def chat_data_ingestion_step(
    data_sources: List[str],
    channels_config: Dict[str, List[str]] = None,
    days_back: int = 1,
    max_messages: Optional[int] = None,
    include_threads: bool = True,
) -> Annotated[RawConversationData, "raw_data"]:
    """Ingest chat data from specified sources.

    Args:
        data_sources: List of sources to fetch from (discord, slack)
        channels_config: Dict mapping source to list of channels
        days_back: Number of days to look back for messages
        max_messages: Max messages per history request (None for unlimited)
        include_threads: When True, fetch Discord thread messages

    Returns:
        RawConversationData: Collected conversations from all sources
    """
    logger.info(f"Starting data ingestion from sources: {data_sources}")

    if channels_config is None:
        # Default channel configuration
        channels_config = {
            "discord": [
                "panagent-team",
            ],
            "slack": [],
        }

    all_conversations = []

    # Discord ingestion
    if "discord" in data_sources:
        discord_token = os.getenv("DISCORD_BOT_TOKEN")
        if not discord_token:
            logger.warning("DISCORD_BOT_TOKEN not found, skipping Discord")
        else:
            try:
                discord_client = DiscordClient(discord_token)
                discord_conversations = asyncio.run(
                    discord_client.fetch_messages(
                        channels=channels_config.get("discord", []),
                        days_back=days_back,
                        max_messages=max_messages,
                        include_threads=include_threads,
                    )
                )
                all_conversations.extend(discord_conversations)
                logger.info(
                    f"Successfully fetched {len(discord_conversations)} Discord conversations"
                )

            except Exception as e:
                logger.error(f"Failed to fetch Discord data: {e}")

    # Slack ingestion
    if "slack" in data_sources:
        slack_token = os.getenv("SLACK_BOT_TOKEN")
        if not slack_token:
            logger.warning("SLACK_BOT_TOKEN not found, skipping Slack")
        else:
            try:
                slack_client = SlackClient(slack_token)
                slack_conversations = slack_client.fetch_messages(
                    channels=channels_config.get("slack", []),
                    days_back=days_back,
                )
                all_conversations.extend(slack_conversations)
                logger.info(
                    f"Successfully fetched {len(slack_conversations)} Slack conversations"
                )

            except Exception as e:
                logger.error(f"Failed to fetch Slack data: {e}")

    # Create raw conversation data
    raw_data = RawConversationData(
        conversations=all_conversations,
        sources=data_sources,
        collection_timestamp=datetime.now(UTC),
        total_conversations=len(all_conversations),
    )

    total_messages = sum(conv.total_messages for conv in all_conversations)
    logger.info(
        f"Data ingestion complete: {len(all_conversations)} conversations, {total_messages} total messages"
    )

    return raw_data
