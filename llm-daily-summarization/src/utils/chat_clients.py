"""
Chat client implementations for Discord and Slack platforms.

This module provides platform-specific clients for fetching messages
from Discord and Slack channels.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import discord
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from zenml.logger import get_logger

from .models import ChatMessage, ConversationData

logger = get_logger(__name__)


class DiscordClient:
    """Discord client for fetching messages and posting summaries."""

    def __init__(self, token: str, default_max_messages: Optional[int] = None):
        """Initialize Discord client.

        Args:
            token: Discord bot token for authentication
            default_max_messages: Default maximum number of messages to
                pull per history request. ``None`` means unlimited.
        """
        self.token = token
        self.client = None
        self.default_max_messages = default_max_messages

    async def fetch_messages(
        self,
        channels: List[str],
        days_back: int = 1,
        *,
        max_messages: Optional[int] = None,
        include_threads: bool = True,
    ) -> List[ConversationData]:
        """Fetch messages (and optionally threads) from Discord.

        Args:
            channels: List of Discord channel names to fetch from
            days_back: Number of days to look back for messages
            max_messages: Max messages to retrieve per history request.
                Overrides the constructor default. ``None`` means unlimited.
            include_threads: When True, fetch messages from active and
                archived threads for each channel.

        Returns:
            List of ConversationData objects containing Discord messages
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True  # needed for channel & thread enumeration
        self.client = discord.Client(intents=intents)

        conversations: List[ConversationData] = []

        # Determine the effective limit to apply to all history() calls
        effective_limit: Optional[int] = (
            max_messages
            if max_messages is not None
            else self.default_max_messages
        )

        @self.client.event
        async def on_ready():
            logger.info(f"Discord client logged in as {self.client.user}")

            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_back)

            for channel_name in channels:
                try:
                    # Find channel by name
                    channel = discord.utils.get(
                        self.client.get_all_channels(), name=channel_name
                    )
                    if not channel:
                        logger.warning(f"Channel '{channel_name}' not found")
                        continue

                    # ------------------------------------------------------------------
                    # 1. Channel messages
                    # ------------------------------------------------------------------
                    messages: List[ChatMessage] = []
                    participants = set()

                    async for message in channel.history(
                        after=cutoff_time, limit=effective_limit
                    ):
                        if message.author.bot:
                            continue  # Skip bot messages

                        chat_message = ChatMessage(
                            id=str(message.id),
                            author=message.author.display_name,
                            content=message.content,
                            timestamp=message.created_at,
                            channel=channel_name,
                            thread_id=None,
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
                        conversations.append(
                            ConversationData(
                                messages=messages,
                                channel_name=channel_name,
                                thread_name=None,
                                source="discord",
                                date_range={
                                    "start": min(
                                        msg.timestamp for msg in messages
                                    ),
                                    "end": max(
                                        msg.timestamp for msg in messages
                                    ),
                                },
                                participant_count=len(participants),
                                total_messages=len(messages),
                            )
                        )
                        logger.info(
                            f"Fetched {len(messages)} channel messages from #{channel_name}"
                        )

                    # ------------------------------------------------------------------
                    # 2. Thread messages (active + archived)
                    # ------------------------------------------------------------------
                    if include_threads and isinstance(
                        channel, discord.TextChannel
                    ):
                        thread_objs = list(channel.threads)

                        # Archived threads are returned via async iterator
                        async for arch_thread in channel.archived_threads(
                            limit=None
                        ):
                            thread_objs.append(arch_thread)

                        for thread in thread_objs:
                            # Respect days_back on thread creation date
                            if thread.created_at < cutoff_time:
                                logger.debug(
                                    f"Skipping thread '{thread.name}' - created at {thread.created_at} is before cutoff {cutoff_time}"
                                )
                                continue

                            thread_messages: List[ChatMessage] = []
                            thread_participants = set()

                            try:
                                async for tmsg in thread.history(
                                    after=cutoff_time, limit=effective_limit
                                ):
                                    if tmsg.author.bot:
                                        continue

                                    thread_messages.append(
                                        ChatMessage(
                                            id=str(tmsg.id),
                                            author=tmsg.author.display_name,
                                            content=tmsg.content,
                                            timestamp=tmsg.created_at,
                                            channel=channel_name,
                                            thread_id=str(thread.id),
                                            source="discord",
                                            metadata={
                                                "message_type": str(tmsg.type),
                                                "edited_at": tmsg.edited_at.isoformat()
                                                if tmsg.edited_at
                                                else None,
                                                "attachments": len(
                                                    tmsg.attachments
                                                ),
                                            },
                                        )
                                    )
                                    thread_participants.add(
                                        tmsg.author.display_name
                                    )

                            except discord.errors.Forbidden:
                                logger.warning(
                                    f"No permission to read thread '{thread.name}' in #{channel_name}"
                                )
                                continue
                            except Exception as e:
                                logger.error(
                                    f"Error fetching thread '{thread.name}' in #{channel_name}: {e}"
                                )
                                continue

                            if thread_messages:
                                conversations.append(
                                    ConversationData(
                                        messages=thread_messages,
                                        channel_name=channel_name,
                                        thread_name=thread.name,
                                        source="discord",
                                        date_range={
                                            "start": min(
                                                m.timestamp
                                                for m in thread_messages
                                            ),
                                            "end": max(
                                                m.timestamp
                                                for m in thread_messages
                                            ),
                                        },
                                        participant_count=len(
                                            thread_participants
                                        ),
                                        total_messages=len(thread_messages),
                                    )
                                )
                                logger.info(
                                    f"Fetched {len(thread_messages)} messages from thread '{thread.name}' in #{channel_name}"
                                )

                            # polite sleep to avoid hitting rate limits
                            await asyncio.sleep(1)

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

    async def post_summary(
        self,
        channel_id: str,
        summary_text: str,
        max_length: int = 1900,
    ) -> bool:
        """Post a summary to a Discord channel, handling message length limits.

        Args:
            channel_id: Discord channel ID where the summary should be posted
            summary_text: The summary text to post
            max_length: Maximum characters per message (default 1900 to leave
                room within Discord's 2000 char limit)

        Returns:
            bool: True if successfully posted, False otherwise
        """
        intents = discord.Intents.default()
        intents.guilds = True
        self.client = discord.Client(intents=intents)

        success = False

        @self.client.event
        async def on_ready():
            nonlocal success
            try:
                channel = await self.client.fetch_channel(int(channel_id))

                # Split the summary into chunks if needed
                chunks = self._split_summary_for_discord(summary_text, max_length)

                for i, chunk in enumerate(chunks):
                    if i > 0:
                        await asyncio.sleep(1)  # Rate limiting between messages
                    await channel.send(chunk, suppress_embeds=True)
                    logger.info(f"Posted chunk {i+1}/{len(chunks)} to Discord channel {channel_id}")

                success = True
                logger.info(f"Successfully posted summary to Discord channel {channel_id} ({len(chunks)} message(s))")

            except Exception as e:
                logger.error(f"Error posting summary to Discord: {e}")
            finally:
                await self.client.close()
                await asyncio.sleep(0.1)  # Small delay to ensure cleanup

        try:
            await self.client.start(self.token)
        except Exception as e:
            logger.error(f"Error during Discord client execution: {e}")

        return success

    def _split_summary_for_discord(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks that fit within Discord's message limit.

        Attempts to split intelligently by sections and bullet points.

        Args:
            text: The text to split
            max_length: Maximum length per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]

        chunks = []
        lines = text.split("\n")

        # Find section boundaries (lines starting with ##)
        sections = []
        current_section = []

        for line in lines:
            if line.startswith("## ") and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        # Try to keep sections together
        current_chunk = sections[0] if sections else ""

        for section in sections[1:]:
            # If adding this section would exceed limit
            if len(current_chunk) + len(section) + 2 > max_length:
                # If the section itself is too long, we need to split it
                if len(section) > max_length:
                    # First, add current chunk if it has content
                    if current_chunk.strip():
                        chunks.append(current_chunk)

                    # Split the long section by bullet points
                    section_lines = section.split("\n")
                    current_chunk = section_lines[0] + "\n"  # Keep header

                    for line in section_lines[1:]:
                        if len(current_chunk) + len(line) + 1 <= max_length:
                            current_chunk += line + "\n"
                        else:
                            chunks.append(current_chunk.rstrip())
                            current_chunk = line + "\n"
                else:
                    # Section fits, so save current chunk and start new one
                    chunks.append(current_chunk)
                    current_chunk = section
            else:
                # Section fits in current chunk
                current_chunk += "\n\n" + section

        if current_chunk.strip():
            chunks.append(current_chunk)

        # Clean up any empty or very short chunks
        return [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 10]


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
            datetime.now(timezone.utc) - timedelta(days=days_back)
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
