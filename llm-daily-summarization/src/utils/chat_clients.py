"""
Chat client implementations for Discord and Slack platforms.

This module provides platform-specific clients for fetching messages
from Discord and Slack channels.
"""

from datetime import datetime, timedelta
from typing import List

import discord
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from zenml.logger import get_logger

from .models import ChatMessage, ConversationData

logger = get_logger(__name__)


class DiscordClient:
    """Discord client for fetching messages."""

    def __init__(self, token: str):
        """Initialize Discord client.

        Args:
            token: Discord bot token for authentication
        """
        self.token = token
        self.client = None

    async def fetch_messages(
        self, channels: List[str], days_back: int = 1
    ) -> List[ConversationData]:
        """Fetch messages from Discord channels.

        Args:
            channels: List of Discord channel names to fetch from
            days_back: Number of days to look back for messages

        Returns:
            List of ConversationData objects containing Discord messages
        """
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)

        conversations = []

        @self.client.event
        async def on_ready():
            logger.info(f"Discord client logged in as {self.client.user}")

            cutoff_time = datetime.utcnow() - timedelta(days=days_back)

            for channel_name in channels:
                try:
                    # Find channel by name
                    channel = discord.utils.get(
                        self.client.get_all_channels(), name=channel_name
                    )
                    if not channel:
                        logger.warning(f"Channel '{channel_name}' not found")
                        continue

                    messages = []
                    participants = set()

                    async for message in channel.history(
                        after=cutoff_time, limit=1000
                    ):
                        if message.author.bot:
                            continue  # Skip bot messages

                        chat_message = ChatMessage(
                            id=str(message.id),
                            author=message.author.display_name,
                            content=message.content,
                            timestamp=message.created_at,
                            channel=channel_name,
                            source="discord",
                            metadata={
                                "message_type": str(message.type),
                                "edited_at": message.edited_at.isoformat()
                                if message.edited_at
                                else None,
                                "attachments": len(message.attachments),
                            },
                        )
                        messages.append(chat_message)
                        participants.add(message.author.display_name)

                    if messages:
                        conversation = ConversationData(
                            messages=messages,
                            channel_name=channel_name,
                            source="discord",
                            date_range={
                                "start": min(
                                    msg.timestamp for msg in messages
                                ),
                                "end": max(msg.timestamp for msg in messages),
                            },
                            participant_count=len(participants),
                            total_messages=len(messages),
                        )
                        conversations.append(conversation)
                        logger.info(
                            f"Fetched {len(messages)} messages from #{channel_name}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error fetching from channel {channel_name}: {e}"
                    )

            await self.client.close()

        try:
            await self.client.start(self.token)
        except Exception as e:
            logger.error(f"Failed to connect to Discord: {e}")
            await self.client.close()

        return conversations


class SlackClient:
    """Slack client for fetching messages."""

    def __init__(self, token: str):
        """Initialize Slack client.

        Args:
            token: Slack bot token for authentication
        """
        self.client = WebClient(token=token)

    def fetch_messages(
        self, channels: List[str], days_back: int = 1
    ) -> List[ConversationData]:
        """Fetch messages from Slack channels.

        Args:
            channels: List of Slack channel names to fetch from
            days_back: Number of days to look back for messages

        Returns:
            List of ConversationData objects containing Slack messages
        """
        conversations = []
        cutoff_timestamp = (
            datetime.utcnow() - timedelta(days=days_back)
        ).timestamp()

        for channel_name in channels:
            try:
                # Get channel ID from name
                channels_response = self.client.conversations_list()
                channel_id = None

                for channel in channels_response["channels"]:
                    if channel["name"] == channel_name:
                        channel_id = channel["id"]
                        break

                if not channel_id:
                    logger.warning(f"Slack channel '{channel_name}' not found")
                    continue

                # Fetch messages
                response = self.client.conversations_history(
                    channel=channel_id,
                    oldest=str(cutoff_timestamp),
                    limit=1000,
                )

                messages = []
                participants = set()

                for msg in response["messages"]:
                    if (
                        msg.get("bot_id")
                        or msg.get("subtype") == "bot_message"
                    ):
                        continue  # Skip bot messages

                    # Get user info
                    user_info = self.client.users_info(user=msg["user"])
                    user_name = (
                        user_info["user"]["display_name"]
                        or user_info["user"]["real_name"]
                    )

                    chat_message = ChatMessage(
                        id=msg["ts"],
                        author=user_name,
                        content=msg["text"],
                        timestamp=datetime.fromtimestamp(float(msg["ts"])),
                        channel=channel_name,
                        source="slack",
                        metadata={
                            "thread_ts": msg.get("thread_ts"),
                            "edited": msg.get("edited"),
                            "reactions": len(msg.get("reactions", [])),
                        },
                    )
                    messages.append(chat_message)
                    participants.add(user_name)

                if messages:
                    conversation = ConversationData(
                        messages=messages,
                        channel_name=channel_name,
                        source="slack",
                        date_range={
                            "start": min(msg.timestamp for msg in messages),
                            "end": max(msg.timestamp for msg in messages),
                        },
                        participant_count=len(participants),
                        total_messages=len(messages),
                    )
                    conversations.append(conversation)
                    logger.info(
                        f"Fetched {len(messages)} messages from #{channel_name}"
                    )

            except SlackApiError as e:
                logger.error(
                    f"Slack API error for channel {channel_name}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Error fetching from Slack channel {channel_name}: {e}"
                )

        return conversations
