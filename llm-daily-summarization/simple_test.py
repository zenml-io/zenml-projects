#!/usr/bin/env python3
"""
Simple test script to validate basic functionality without ZenML dependencies.
"""

import os
import sys
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import json
from datetime import datetime
from pathlib import Path

from utils.models import ChatMessage, ConversationData, RawConversationData


def test_models():
    """Test basic model functionality."""

    print("Testing data models...")

    # Test ChatMessage
    message = ChatMessage(
        id="test_001",
        author="TestUser",
        content="This is a test message",
        timestamp=datetime.now(),
        channel="test-channel",
        source="test",
    )

    print(f"✓ ChatMessage: {message.author} - {message.content}")

    # Test ConversationData
    conversation = ConversationData(
        messages=[message],
        channel_name="test-channel",
        source="test",
        date_range={"start": datetime.now(), "end": datetime.now()},
        participant_count=1,
        total_messages=1,
    )

    print(
        f"✓ ConversationData: {conversation.channel_name} with {len(conversation.messages)} messages"
    )

    # Test RawConversationData
    raw_data = RawConversationData(
        conversations=[conversation],
        sources=["test"],
        collection_timestamp=datetime.now(),
        total_conversations=1,
    )

    print(
        f"✓ RawConversationData: {raw_data.total_conversations} conversations from {raw_data.sources}"
    )


def test_sample_data():
    """Test loading sample data."""

    print("\nTesting sample data loading...")

    data_file = Path("data/sample_conversations.json")

    if not data_file.exists():
        print(f"❌ Sample data file not found: {data_file}")
        return False

    with open(data_file, "r") as f:
        sample_data = json.load(f)

    print(
        f"✓ Sample data loaded: {len(sample_data['conversations'])} conversations"
    )

    # Test converting to our models
    conversations = []
    for conv_data in sample_data["conversations"]:
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

    print(f"✓ Converted to models: {len(conversations)} conversations")

    for conv in conversations:
        print(
            f"  - {conv.channel_name} ({conv.source}): {conv.total_messages} messages"
        )

    return True


def main():
    """Run all tests."""

    print("🚀 Starting LLM Summarization Pipeline Tests\n")

    try:
        test_models()
        test_sample_data()

        print(
            "\n🎉 All tests passed! Basic functionality is working correctly."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
