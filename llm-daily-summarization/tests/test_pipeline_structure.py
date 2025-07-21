"""
Basic tests to validate pipeline structure and individual components.
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

# Import our models and steps
from src.utils.models import (
    ChatMessage, ConversationData, RawConversationData, 
    CleanedConversationData, TaskItem, Summary
)
from src.steps.mock_data_ingestion import mock_chat_data_ingestion_step
from src.steps.preprocessing import text_preprocessing_step
from src.agents.summarizer_agent import SummarizerAgent
from src.agents.task_extractor_agent import TaskExtractorAgent


class TestDataModels:
    """Test our Pydantic data models."""
    
    def test_chat_message_creation(self):
        """Test ChatMessage model creation."""
        
        message = ChatMessage(
            id="test_001",
            author="TestUser",
            content="This is a test message",
            timestamp=datetime.now(),
            channel="test-channel",
            source="test",
            metadata={"test": True}
        )
        
        assert message.id == "test_001"
        assert message.author == "TestUser"
        assert message.source == "test"
    
    def test_conversation_data_creation(self):
        """Test ConversationData model creation."""
        
        messages = [
            ChatMessage(
                id="msg_1",
                author="User1",
                content="Hello",
                timestamp=datetime.now(),
                channel="test",
                source="test"
            )
        ]
        
        conversation = ConversationData(
            messages=messages,
            channel_name="test-channel",
            source="test",
            date_range={"start": datetime.now(), "end": datetime.now()},
            participant_count=1,
            total_messages=1
        )
        
        assert len(conversation.messages) == 1
        assert conversation.channel_name == "test-channel"
        assert conversation.participant_count == 1


class TestMockDataIngestion:
    """Test the mock data ingestion step."""
    
    def test_mock_data_ingestion(self):
        """Test that mock data ingestion works."""
        
        # Test with discord source
        result = mock_chat_data_ingestion_step(
            data_sources=["discord"],
            sample_data_path="data/sample_conversations.json"
        )
        
        assert isinstance(result, RawConversationData)
        assert len(result.conversations) >= 1
        assert "discord" in result.sources
        
        # Check that we have valid conversation data
        for conversation in result.conversations:
            assert len(conversation.messages) > 0
            assert conversation.source in ["discord", "slack"]
    
    def test_mock_data_ingestion_multiple_sources(self):
        """Test mock data ingestion with multiple sources."""
        
        result = mock_chat_data_ingestion_step(
            data_sources=["discord", "slack"],
            sample_data_path="data/sample_conversations.json"
        )
        
        assert isinstance(result, RawConversationData)
        assert len(result.conversations) >= 2
        
        # Should have conversations from both sources
        sources = {conv.source for conv in result.conversations}
        assert "discord" in sources
        assert "slack" in sources


class TestTextPreprocessing:
    """Test the text preprocessing step."""
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality."""
        
        # First get some raw data
        raw_data = mock_chat_data_ingestion_step(
            data_sources=["discord"],
            sample_data_path="data/sample_conversations.json"
        )
        
        # Preprocess it
        cleaned_data = text_preprocessing_step(raw_data=raw_data)
        
        assert isinstance(cleaned_data, CleanedConversationData)
        assert len(cleaned_data.conversations) > 0
        assert cleaned_data.word_count > 0
        
        # Check that messages were actually cleaned
        for conversation in cleaned_data.conversations:
            for message in conversation.messages:
                # Content should be non-empty and cleaned
                assert len(message.content.strip()) > 0
                # Should not contain raw URLs (they should be replaced with [URL])
                assert "http://" not in message.content
                assert "https://" not in message.content


class TestAgents:
    """Test the LangGraph agents (without actual LLM calls)."""
    
    def test_summarizer_agent_initialization(self):
        """Test that summarizer agent can be initialized."""
        
        model_config = {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        # This should not fail even without actual credentials
        # since we're just testing initialization
        try:
            agent = SummarizerAgent(model_config)
            assert agent.model_config == model_config
        except Exception as e:
            # If it fails due to missing credentials, that's expected
            assert "credentials" in str(e).lower() or "authentication" in str(e).lower()
    
    def test_task_extractor_agent_initialization(self):
        """Test that task extractor agent can be initialized."""
        
        model_config = {
            "model_name": "gemini-2.5-flash",
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        try:
            agent = TaskExtractorAgent(model_config)
            assert agent.model_config == model_config
            assert len(agent.task_indicators) > 0
        except Exception as e:
            # If it fails due to missing credentials, that's expected
            assert "credentials" in str(e).lower() or "authentication" in str(e).lower()


class TestDataFlow:
    """Test the overall data flow through pipeline steps."""
    
    def test_mock_to_preprocessing_flow(self):
        """Test data flow from mock ingestion to preprocessing."""
        
        # Step 1: Mock data ingestion
        raw_data = mock_chat_data_ingestion_step(
            data_sources=["discord", "slack"],
            sample_data_path="data/sample_conversations.json"
        )
        
        # Step 2: Preprocessing
        cleaned_data = text_preprocessing_step(raw_data=raw_data)
        
        # Validate data flow
        assert isinstance(raw_data, RawConversationData)
        assert isinstance(cleaned_data, CleanedConversationData)
        
        # Should have preserved conversation structure
        assert len(cleaned_data.conversations) > 0
        
        # Word count should be calculated
        assert cleaned_data.word_count > 0
        
        # Processing notes should be present
        assert len(cleaned_data.processing_notes) > 0
    
    def test_conversation_data_consistency(self):
        """Test that conversation data remains consistent through transformations."""
        
        raw_data = mock_chat_data_ingestion_step(
            data_sources=["discord"],
            sample_data_path="data/sample_conversations.json"
        )
        
        cleaned_data = text_preprocessing_step(raw_data=raw_data)
        
        # Check consistency
        for raw_conv, cleaned_conv in zip(raw_data.conversations, cleaned_data.conversations):
            assert raw_conv.channel_name == cleaned_conv.channel_name
            assert raw_conv.source == cleaned_conv.source
            # Message count might be different due to filtering
            assert cleaned_conv.total_messages <= raw_conv.total_messages


# Integration test that requires actual LLM access
@pytest.mark.integration
class TestWithLLM:
    """Integration tests that require actual LLM access."""
    
    def test_full_pipeline_flow(self):
        """Test the full pipeline flow (requires LLM credentials)."""
        
        # This test would require actual Vertex AI credentials
        # and would test the full pipeline including LangGraph agents
        pytest.skip("Requires LLM credentials - run manually with proper setup")


if __name__ == "__main__":
    # Run basic tests
    test_models = TestDataModels()
    test_models.test_chat_message_creation()
    test_models.test_conversation_data_creation()
    print("✓ Data models tests passed")
    
    test_ingestion = TestMockDataIngestion()
    test_ingestion.test_mock_data_ingestion()
    test_ingestion.test_mock_data_ingestion_multiple_sources()
    print("✓ Mock data ingestion tests passed")
    
    test_preprocessing = TestTextPreprocessing()
    test_preprocessing.test_text_preprocessing()
    print("✓ Text preprocessing tests passed")
    
    test_agents = TestAgents()
    test_agents.test_summarizer_agent_initialization()
    test_agents.test_task_extractor_agent_initialization()
    print("✓ Agent initialization tests passed")
    
    test_flow = TestDataFlow()
    test_flow.test_mock_to_preprocessing_flow()
    test_flow.test_conversation_data_consistency()
    print("✓ Data flow tests passed")
    
    print("\n🎉 All basic tests passed! The pipeline structure is working correctly.")