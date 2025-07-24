"""
Mock data ingestion step for testing the pipeline without external API dependencies.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

from ..utils.models import ChatMessage, ConversationData, RawConversationData

logger = get_logger(__name__)


@step
def mock_chat_data_ingestion_step(
    data_sources: List[str],
    sample_data_path: str = "data/sample_conversations.json",
) -> Annotated[RawConversationData, "raw_data"]:
    """
    Mock data ingestion step using sample data for testing.

    Args:
        data_sources: List of sources to simulate (discord, slack)
        sample_data_path: Path to sample data file

    Returns:
        RawConversationData: Mock conversation data
    """

    logger.info(f"Starting mock data ingestion from sources: {data_sources}")

    # Load sample data
    project_root = Path(__file__).parent.parent.parent
    data_file = project_root / sample_data_path

    if not data_file.exists():
        logger.error(f"Sample data file not found: {data_file}")
        raise FileNotFoundError(f"Sample data file not found: {data_file}")

    with open(data_file, "r") as f:
        sample_data = json.load(f)

    # Convert to our data models
    conversations = []

    for conv_data in sample_data["conversations"]:
        # Filter by requested sources
        if conv_data["source"] not in data_sources:
            continue

        # Convert messages
        messages = []
        for msg_data in conv_data["messages"]:
            message = ChatMessage(
                id=msg_data["id"],
                author=msg_data["author"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(
                    msg_data["timestamp"].replace("Z", "+00:00")
                ),
                channel=msg_data["channel"],
                source=msg_data["source"],
                metadata=msg_data.get("metadata", {}),
            )
            messages.append(message)

        # Create conversation
        conversation = ConversationData(
            messages=messages,
            channel_name=conv_data["channel_name"],
            source=conv_data["source"],
            date_range={
                "start": datetime.fromisoformat(
                    conv_data["date_range"]["start"].replace("Z", "+00:00")
                ),
                "end": datetime.fromisoformat(
                    conv_data["date_range"]["end"].replace("Z", "+00:00")
                ),
            },
            participant_count=conv_data["participant_count"],
            total_messages=conv_data["total_messages"],
        )
        conversations.append(conversation)

    # Create raw conversation data
    raw_data = RawConversationData(
        conversations=conversations,
        sources=data_sources,
        collection_timestamp=datetime.utcnow(),
        total_conversations=len(conversations),
    )

    total_messages = sum(conv.total_messages for conv in conversations)
    logger.info(
        f"Mock data ingestion complete: {len(conversations)} conversations, {total_messages} total messages"
    )

    return raw_data
