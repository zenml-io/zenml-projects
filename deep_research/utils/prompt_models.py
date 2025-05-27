"""Pydantic models for prompt tracking and management.

This module contains models for bundling prompts as trackable artifacts
in the ZenML pipeline, enabling better observability and version control.
"""

from datetime import datetime
from typing import Dict, Optional

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


class PromptsBundle(BaseModel):
    """Bundle of all prompts used in the research pipeline.

    This model serves as a single artifact that contains all prompts,
    making them trackable, versionable, and visualizable in the ZenML dashboard.
    """

    # Core prompts used in the pipeline
    search_query_prompt: PromptTemplate
    query_decomposition_prompt: PromptTemplate
    synthesis_prompt: PromptTemplate
    viewpoint_analysis_prompt: PromptTemplate
    reflection_prompt: PromptTemplate
    additional_synthesis_prompt: PromptTemplate
    conclusion_generation_prompt: PromptTemplate
    executive_summary_prompt: PromptTemplate
    introduction_prompt: PromptTemplate

    # Metadata
    pipeline_version: str = Field(
        "1.0.0", description="Version of the pipeline using these prompts"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when this bundle was created",
    )

    # Additional prompts can be stored here
    custom_prompts: Dict[str, PromptTemplate] = Field(
        default_factory=dict,
        description="Additional custom prompts not part of the core set",
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }

    def get_prompt_by_name(self, name: str) -> Optional[PromptTemplate]:
        """Retrieve a prompt by its name.

        Args:
            name: Name of the prompt to retrieve

        Returns:
            PromptTemplate if found, None otherwise
        """
        # Check core prompts
        for field_name, field_value in self.__dict__.items():
            if (
                isinstance(field_value, PromptTemplate)
                and field_value.name == name
            ):
                return field_value

        # Check custom prompts
        return self.custom_prompts.get(name)

    def list_all_prompts(self) -> Dict[str, PromptTemplate]:
        """Get all prompts as a dictionary.

        Returns:
            Dictionary mapping prompt names to PromptTemplate objects
        """
        all_prompts = {}

        # Add core prompts
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, PromptTemplate):
                all_prompts[field_value.name] = field_value

        # Add custom prompts
        all_prompts.update(self.custom_prompts)

        return all_prompts

    def get_prompt_content(self, prompt_type: str) -> str:
        """Get the content of a specific prompt by its type.

        Args:
            prompt_type: Type of prompt (e.g., 'search_query_prompt', 'synthesis_prompt')

        Returns:
            The prompt content string

        Raises:
            AttributeError: If prompt type doesn't exist
        """
        prompt = getattr(self, prompt_type)
        return prompt.content
