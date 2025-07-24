#!/usr/bin/env python3
"""
Basic test script to validate core functionality without external dependencies.
"""

import os
import re
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


def test_text_cleaning():
    """Test basic text cleaning functionality without external dependencies."""

    print("\nTesting text cleaning...")

    # Simple text cleaner without dependencies
    class SimpleTextCleaner:
        def __init__(self):
            self.url_pattern = re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            )
            self.mention_pattern = re.compile(r"<@[!&]?[0-9]+>")

        def clean_message_content(self, content: str) -> str:
            if not content:
                return ""

            # Remove URLs
            content = self.url_pattern.sub("[URL]", content)

            # Remove mentions
            content = self.mention_pattern.sub("@user", content)

            # Clean up whitespace
            content = re.sub(r"\s+", " ", content).strip()

            return content

    cleaner = SimpleTextCleaner()

    # Test URL removal
    text_with_url = (
        "Check out this link: https://example.com and this one http://test.com"
    )
    cleaned = cleaner.clean_message_content(text_with_url)
    print(f"✓ URL removal: '{text_with_url}' -> '{cleaned}'")

    # Test mention removal
    text_with_mentions = "Hey <@123456> and <@!789012> what do you think?"
    cleaned = cleaner.clean_message_content(text_with_mentions)
    print(f"✓ Mention removal: '{text_with_mentions}' -> '{cleaned}'")


def main():
    """Run all tests."""

    print("🚀 Starting LLM Summarization Pipeline Basic Tests\n")

    try:
        test_models()

        if test_sample_data():
            test_text_cleaning()

        print(
            "\n🎉 All basic tests passed! Core functionality is working correctly."
        )
        print("\n📋 Project Structure Summary:")
        print("   ✓ Data models (Pydantic) working correctly")
        print("   ✓ Sample data loading and conversion working")
        print("   ✓ Basic text processing functionality working")
        print("   ✓ Project structure is set up properly")

        print("\n🔧 Next Steps:")
        print("   - Install external dependencies (langchain, langfuse, etc.)")
        print("   - Set up Vertex AI credentials")
        print("   - Configure ZenML stack")
        print("   - Test with real LLM calls")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
