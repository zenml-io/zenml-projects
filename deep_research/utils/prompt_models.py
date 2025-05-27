"""Pydantic models for prompt tracking and management.

This module contains models for tracking prompts as artifacts
in the ZenML pipeline, enabling better observability and version control.
"""

from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    """Represents a single prompt template with metadata."""

    name: str = Field(..., description="Unique identifier for the prompt")
    content: str = Field(..., description="The actual prompt template content")
    description: str = Field(
        "", description="Human-readable description of what this prompt does"
    )
    version: str = Field("1.0.0", description="Version of the prompt template")
    tags: list[str] = Field(
        default_factory=list, description="Tags for categorizing prompts"
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }
