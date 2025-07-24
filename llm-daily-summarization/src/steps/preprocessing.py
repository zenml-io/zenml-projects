"""
Text preprocessing step for cleaning and preparing conversation data.
"""

import re

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from ..utils.models import (
    ChatMessage,
    CleanedConversationData,
    ConversationData,
    RawConversationData,
)

logger = get_logger(__name__)


class TextCleaner:
    """Text cleaning utilities."""

    def __init__(self):
        # Common patterns to remove or clean
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self.mention_pattern = re.compile(
            r"<@[!&]?[0-9]+>"
        )  # Discord mentions
        self.slack_mention_pattern = re.compile(
            r"<@U[A-Z0-9]+>"
        )  # Slack user mentions
        self.channel_mention_pattern = re.compile(
            r"<#[0-9]+>"
        )  # Discord channel mentions
        self.emoji_pattern = re.compile(r":[a-zA-Z0-9_]+:")  # Custom emoji
        self.code_block_pattern = re.compile(
            r"```.*?```", re.DOTALL
        )  # Code blocks
        self.inline_code_pattern = re.compile(r"`[^`]+`")  # Inline code

        # Bot patterns and system messages to filter out
        self.bot_indicators = [
            "joined the server",
            "left the server",
            "pinned a message",
            "started a call",
            "ended a call",
            "changed their nickname",
            "uploaded a file",
        ]

    def clean_message_content(self, content: str) -> str:
        """Clean individual message content."""

        if not content or not isinstance(content, str):
            return ""

        # NOTE: URLs are now preserved.
        # Remove mentions (keep readable placeholder)
        content = self.mention_pattern.sub("@user", content)
        content = self.slack_mention_pattern.sub("@user", content)
        content = self.channel_mention_pattern.sub("#channel", content)

        # Handle code blocks (preserve but simplify)
        content = self.code_block_pattern.sub("[CODE_BLOCK]", content)
        content = self.inline_code_pattern.sub("[CODE]", content)

        # NOTE: Custom emojis like :smile: are now preserved.

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content

    def is_system_message(self, message: ChatMessage) -> bool:
        """Check if message is a system/bot message."""

        content_lower = message.content.lower()

        # Check for bot indicators
        for indicator in self.bot_indicators:
            if indicator in content_lower:
                return True

        # Removed minimum length filter to allow very short messages
        # (previously filtered out messages shorter than 3 characters)

        # Check if message is mostly special characters
        special_char_ratio = sum(
            1 for c in message.content if not c.isalnum() and not c.isspace()
        ) / max(len(message.content), 1)
        if special_char_ratio > 0.7:
            return True

        return False

    def should_keep_message(self, message: ChatMessage) -> bool:
        """Determine if message should be kept after cleaning."""

        # Filter out system messages
        if self.is_system_message(message):
            return False

        # Clean the content and check if meaningful content remains
        cleaned_content = self.clean_message_content(message.content)

        # Keep if cleaned content has meaningful text
        return len(cleaned_content.strip()) >= 1


@step
def text_preprocessing_step(
    raw_data: RawConversationData,
    min_message_length: int = 1,
    max_messages_per_conversation: int = 500,
) -> Annotated[CleanedConversationData, "cleaned_data"]:
    """
    Preprocess and clean conversation data.

    Args:
        raw_data: Raw conversation data from ingestion
        min_message_length: Minimum message length to keep
        max_messages_per_conversation: Maximum messages per conversation

    Returns:
        CleanedConversationData: Cleaned and processed conversation data
    """

    logger.info(
        f"Starting text preprocessing for {len(raw_data.conversations)} conversations"
    )

    cleaner = TextCleaner()
    cleaned_conversations = []
    total_removed_messages = 0
    processing_notes = []

    for conversation in raw_data.conversations:
        logger.info(
            f"Processing conversation: {conversation.channel_name} ({conversation.source})"
        )

        # Filter and clean messages
        cleaned_messages = []
        removed_count = 0

        for message in conversation.messages:
            if cleaner.should_keep_message(message):
                # Clean the message content
                cleaned_content = cleaner.clean_message_content(
                    message.content
                )

                if len(cleaned_content) >= min_message_length:
                    cleaned_message = ChatMessage(
                        id=message.id,
                        author=message.author,
                        content=cleaned_content,
                        timestamp=message.timestamp,
                        channel=message.channel,
                        source=message.source,
                        metadata=message.metadata,
                    )
                    cleaned_messages.append(cleaned_message)
                else:
                    removed_count += 1
            else:
                removed_count += 1

        # Limit messages per conversation if too many
        if len(cleaned_messages) > max_messages_per_conversation:
            # Keep the most recent messages
            cleaned_messages = sorted(
                cleaned_messages, key=lambda x: x.timestamp
            )[-max_messages_per_conversation:]
            removed_count += (
                len(conversation.messages) - max_messages_per_conversation
            )
            processing_notes.append(
                f"Truncated {conversation.channel_name} to {max_messages_per_conversation} messages"
            )

        total_removed_messages += removed_count

        # Create cleaned conversation if any messages remain
        if cleaned_messages:
            # Recalculate participant count
            participants = set(msg.author for msg in cleaned_messages)

            cleaned_conversation = ConversationData(
                messages=cleaned_messages,
                channel_name=conversation.channel_name,
                source=conversation.source,
                date_range={
                    "start": min(msg.timestamp for msg in cleaned_messages),
                    "end": max(msg.timestamp for msg in cleaned_messages),
                },
                participant_count=len(participants),
                total_messages=len(cleaned_messages),
            )
            cleaned_conversations.append(cleaned_conversation)

            logger.info(
                f"Cleaned {conversation.channel_name}: {len(cleaned_messages)} messages kept, {removed_count} removed"
            )
        else:
            processing_notes.append(
                f"Conversation {conversation.channel_name} removed - no valid messages"
            )
            logger.warning(
                f"No valid messages remaining in {conversation.channel_name}"
            )

    # Calculate total word count
    total_word_count = 0
    for conversation in cleaned_conversations:
        for message in conversation.messages:
            total_word_count += len(message.content.split())

    # Add processing summary
    processing_notes.extend(
        [
            f"Processed {len(raw_data.conversations)} conversations",
            f"Kept {len(cleaned_conversations)} conversations",
            f"Removed {total_removed_messages} messages total",
            f"Total word count: {total_word_count}",
        ]
    )

    cleaned_data = CleanedConversationData(
        conversations=cleaned_conversations,
        removed_messages_count=total_removed_messages,
        processing_notes=processing_notes,
        word_count=total_word_count,
    )

    logger.info(
        f"Text preprocessing complete: {len(cleaned_conversations)} conversations, {total_word_count} words"
    )

    return cleaned_data
