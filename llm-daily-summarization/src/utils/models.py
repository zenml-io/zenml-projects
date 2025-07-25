"""
Data models for the LLM Summarization Pipeline.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual chat message model."""

    id: str = Field(description="Unique message identifier")
    author: str = Field(description="Message author name")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(description="Message timestamp")
    channel: str = Field(description="Channel or conversation name")
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread identifier if the message belongs to a Discord thread",
    )
    source: str = Field(description="Source platform (discord, slack)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ConversationData(BaseModel):
    """Collection of chat messages from a conversation."""

    messages: List[ChatMessage] = Field(description="List of chat messages")
    channel_name: str = Field(description="Channel or conversation name")
    thread_name: Optional[str] = Field(
        default=None,
        description="Name of the Discord thread if applicable",
    )
    source: str = Field(description="Source platform")
    date_range: Dict[str, datetime] = Field(
        description="Start and end timestamps"
    )
    participant_count: int = Field(description="Number of unique participants")
    total_messages: int = Field(description="Total number of messages")


class RawConversationData(BaseModel):
    """Raw data from multiple conversation sources."""

    conversations: List[ConversationData] = Field(
        description="List of conversations"
    )
    sources: List[str] = Field(description="Data sources used")
    collection_timestamp: datetime = Field(
        description="When data was collected"
    )
    total_conversations: int = Field(
        description="Total number of conversations"
    )


class CleanedConversationData(BaseModel):
    """Preprocessed and cleaned conversation data."""

    conversations: List[ConversationData] = Field(
        description="Cleaned conversations"
    )
    removed_messages_count: int = Field(
        description="Number of messages removed during cleaning"
    )
    processing_notes: List[str] = Field(description="Notes from preprocessing")
    word_count: int = Field(description="Total word count after cleaning")


class TaskItem(BaseModel):
    """Individual task or action item."""

    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    assignee: Optional[str] = Field(
        default=None, description="Assigned person"
    )
    priority: str = Field(default="medium", description="Priority level")
    due_date: Optional[datetime] = Field(default=None, description="Due date")
    source_messages: List[str] = Field(description="Source message IDs")
    confidence_score: float = Field(description="Extraction confidence score")


class Summary(BaseModel):
    """Conversation summary."""

    title: str = Field(description="Summary title")
    content: str = Field(description="Summary content")
    key_points: List[str] = Field(description="Key discussion points")
    participants: List[str] = Field(description="Key participants")
    topics: List[str] = Field(description="Main topics discussed")
    word_count: int = Field(description="Summary word count")
    confidence_score: float = Field(description="Summary quality score")


class ProcessedData(BaseModel):
    """Output from LangGraph agent processing."""

    summaries: List[Summary] = Field(description="Generated summaries")
    tasks: List[TaskItem] = Field(description="Extracted tasks")
    processing_metadata: Dict[str, Any] = Field(
        description="Processing metadata"
    )
    llm_usage_stats: Dict[str, Any] = Field(description="LLM usage statistics")
    agent_trace_id: str = Field(description="LangGraph trace identifier")
    run_id: str = Field(
        description="ZenML pipeline run identifier for trace tagging"
    )
    errors: List[str] = Field(  # New
        default_factory=list,
        description="Non-fatal errors encountered during processing",
    )


class DeliveryResult(BaseModel):
    """Result from output delivery."""

    target: str = Field(description="Delivery target (notion, slack, github)")
    success: bool = Field(description="Delivery success status")
    delivered_items: List[str] = Field(
        description="Successfully delivered item IDs"
    )
    failed_items: List[str] = Field(description="Failed delivery item IDs")
    delivery_url: Optional[str] = Field(
        default=None, description="URL of delivered content"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


class EvaluationMetrics(BaseModel):
    """Pipeline evaluation metrics."""

    summary_quality_score: float = Field(description="Summary quality rating")
    task_extraction_accuracy: float = Field(
        description="Task extraction accuracy"
    )
    processing_time_seconds: float = Field(description="Total processing time")
    token_usage: Dict[str, int] = Field(description="Token usage statistics")
    cost_estimate: float = Field(description="Estimated cost in USD")
    delivery_success_rate: float = Field(description="Delivery success rate")
    human_feedback_score: Optional[float] = Field(
        default=None, description="Human feedback rating"
    )


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    data_sources: List[str] = Field(description="Data sources to process")
    output_targets: List[str] = Field(description="Output delivery targets")
    llm_config: Dict[str, Any] = Field(description="LLM model configuration")
    processing_config: Dict[str, Any] = Field(
        description="Processing configuration"
    )
    monitoring_config: Dict[str, Any] = Field(
        description="Monitoring configuration"
    )
