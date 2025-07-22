"""
Data ingestion step for chat platforms (Discord, Slack).
"""

import asyncio
import os
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List
from typing_extensions import Annotated

import discord
from langfuse import observe
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from zenml import step
from zenml.logger import get_logger

from ..utils.models import ChatMessage, ConversationData, RawConversationData

logger = get_logger(__name__)


class DiscordClient:
    """Discord client for fetching messages."""
    
    def __init__(self, token: str):
        """Initialize Discord client."""
        self.token = token
        self.client = None
    
    @observe(as_type="retrieval")
    async def fetch_messages(self, channels: List[str], days_back: int = 1) -> List[ConversationData]:
        """Fetch messages from Discord channels."""
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
                    channel = discord.utils.get(self.client.get_all_channels(), name=channel_name)
                    if not channel:
                        logger.warning(f"Channel '{channel_name}' not found")
                        continue
                    
                    messages = []
                    participants = set()
                    
                    async for message in channel.history(after=cutoff_time, limit=1000):
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
                                "edited_at": message.edited_at.isoformat() if message.edited_at else None,
                                "attachments": len(message.attachments)
                            }
                        )
                        messages.append(chat_message)
                        participants.add(message.author.display_name)
                    
                    if messages:
                        conversation = ConversationData(
                            messages=messages,
                            channel_name=channel_name,
                            source="discord",
                            date_range={
                                "start": min(msg.timestamp for msg in messages),
                                "end": max(msg.timestamp for msg in messages)
                            },
                            participant_count=len(participants),
                            total_messages=len(messages)
                        )
                        conversations.append(conversation)
                        logger.info(f"Fetched {len(messages)} messages from #{channel_name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching from channel {channel_name}: {e}")
            
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
        """Initialize Slack client."""
        self.client = WebClient(token=token)
    
    @observe(as_type="retrieval")
    def fetch_messages(self, channels: List[str], days_back: int = 1) -> List[ConversationData]:
        """Fetch messages from Slack channels."""
        conversations = []
        cutoff_timestamp = (datetime.utcnow() - timedelta(days=days_back)).timestamp()
        
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
                    limit=1000
                )
                
                messages = []
                participants = set()
                
                for msg in response["messages"]:
                    if msg.get("bot_id") or msg.get("subtype") == "bot_message":
                        continue  # Skip bot messages
                    
                    # Get user info
                    user_info = self.client.users_info(user=msg["user"])
                    user_name = user_info["user"]["display_name"] or user_info["user"]["real_name"]
                    
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
                            "reactions": len(msg.get("reactions", []))
                        }
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
                            "end": max(msg.timestamp for msg in messages)
                        },
                        participant_count=len(participants),
                        total_messages=len(messages)
                    )
                    conversations.append(conversation)
                    logger.info(f"Fetched {len(messages)} messages from #{channel_name}")
                
            except SlackApiError as e:
                logger.error(f"Slack API error for channel {channel_name}: {e}")
            except Exception as e:
                logger.error(f"Error fetching from Slack channel {channel_name}: {e}")
        
        return conversations


@step
def chat_data_ingestion_step(
    data_sources: List[str],
    channels_config: Dict[str, List[str]] = None,
    days_back: int = 1
) -> Annotated[RawConversationData, "raw_data"]:
    """Ingest chat data from specified sources.
    
    Args:
        data_sources: List of sources to fetch from (discord, slack)
        channels_config: Dict mapping source to list of channels
        days_back: Number of days to look back for messages
        
    Returns:
        RawConversationData: Collected conversations from all sources
    """
    logger.info(f"Starting data ingestion from sources: {data_sources}")
    
    if channels_config is None:
        # Default channel configuration
        channels_config = {
            "discord": ["panagent-team"],
            "slack": []
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
                        days_back=days_back
                    )
                )
                all_conversations.extend(discord_conversations)
                logger.info(f"Successfully fetched {len(discord_conversations)} Discord conversations")
                
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
                    days_back=days_back
                )
                all_conversations.extend(slack_conversations)
                logger.info(f"Successfully fetched {len(slack_conversations)} Slack conversations")
                
            except Exception as e:
                logger.error(f"Failed to fetch Slack data: {e}")
    
    # Create raw conversation data
    raw_data = RawConversationData(
        conversations=all_conversations,
        sources=data_sources,
        collection_timestamp=datetime.now(UTC),
        total_conversations=len(all_conversations)
    )
    
    total_messages = sum(conv.total_messages for conv in all_conversations)
    logger.info(f"Data ingestion complete: {len(all_conversations)} conversations, {total_messages} total messages")
    
    return raw_data