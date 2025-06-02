"""Pydantic model definitions for the research pipeline.

This module contains all the Pydantic models that represent the state of the research
pipeline. These models replace the previous dataclasses implementation and leverage
Pydantic's validation, serialization, and integration with ZenML.
"""

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal


class Prompt(BaseModel):
    """A single prompt with metadata for tracking and visualization.

    This class is designed to be simple and intuitive to use. You can access
    the prompt content directly via the content attribute or by converting
    to string.
    """

    content: str = Field(..., description="The actual prompt text")
    name: str = Field(..., description="Unique identifier for the prompt")
    description: str = Field(
        "", description="Human-readable description of what this prompt does"
    )
    version: str = Field("1.0.0", description="Version of the prompt")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing the prompt"
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }

    def __str__(self) -> str:
        """Return the prompt content as a string."""
        return self.content

    def __repr__(self) -> str:
        """Return a readable representation of the prompt."""
        return f"Prompt(name='{self.name}', version='{self.version}')"

    @classmethod
    def create(
        cls,
        content: str,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
    ) -> "Prompt":
        """Factory method to create a Prompt instance.

        Args:
            content: The prompt text
            name: Unique identifier for the prompt
            description: Optional description of the prompt's purpose
            version: Version string (defaults to "1.0.0")
            tags: Optional list of tags for categorization

        Returns:
            A new Prompt instance
        """
        return cls(
            content=content,
            name=name,
            description=description,
            version=version,
            tags=tags or [],
        )


class SearchResult(BaseModel):
    """Represents a search result for a sub-question."""

    url: str = ""
    content: str = ""
    title: str = ""
    snippet: str = ""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    model_config = {
        "extra": "ignore",  # Ignore extra fields during deserialization
        "frozen": False,  # Allow attribute updates
        "validate_assignment": True,  # Validate when attributes are set
    }


class ViewpointTension(BaseModel):
    """Represents a tension between different viewpoints on a topic."""

    topic: str = ""
    viewpoints: Dict[str, str] = Field(default_factory=dict)

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class SynthesizedInfo(BaseModel):
    """Represents synthesized information for a sub-question."""

    synthesized_answer: str = ""
    key_sources: List[str] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"] = "medium"
    information_gaps: str = ""
    improvements: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class ViewpointAnalysis(BaseModel):
    """Represents the analysis of different viewpoints on the research topic."""

    main_points_of_agreement: List[str] = Field(default_factory=list)
    areas_of_tension: List[ViewpointTension] = Field(default_factory=list)
    perspective_gaps: str = ""
    integrative_insights: str = ""

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class ReflectionMetadata(BaseModel):
    """Metadata about the reflection process."""

    critique_summary: List[str] = Field(default_factory=list)
    additional_questions_identified: List[str] = Field(default_factory=list)
    searches_performed: List[str] = Field(default_factory=list)
    improvements_made: float = Field(
        default=0
    )  # Changed from int to float to handle timestamp values
    error: Optional[str] = None

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class ResearchState(BaseModel):
    """Comprehensive state object for the enhanced research pipeline."""

    # Initial query information
    main_query: str = ""
    sub_questions: List[str] = Field(default_factory=list)

    # Information gathering results
    search_results: Dict[str, List[SearchResult]] = Field(default_factory=dict)

    # Synthesized information
    synthesized_info: Dict[str, SynthesizedInfo] = Field(default_factory=dict)

    # Viewpoint analysis
    viewpoint_analysis: Optional[ViewpointAnalysis] = None

    # Reflection results
    enhanced_info: Dict[str, SynthesizedInfo] = Field(default_factory=dict)
    reflection_metadata: Optional[ReflectionMetadata] = None

    # Final report
    final_report_html: str = ""

    # Search cost tracking
    search_costs: Dict[str, float] = Field(
        default_factory=dict,
        description="Total costs by search provider (e.g., {'exa': 0.0, 'tavily': 0.0})",
    )
    search_cost_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed log of each search with cost information",
    )
    # Format: [{"provider": "exa", "query": "...", "cost": 0.0, "timestamp": ..., "step": "...", "sub_question": "..."}]

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }

    def get_current_stage(self) -> str:
        """Determine the current stage of research based on filled data."""
        if self.final_report_html:
            return "final_report"
        elif self.enhanced_info:
            return "after_reflection"
        elif self.viewpoint_analysis:
            return "after_viewpoint_analysis"
        elif self.synthesized_info:
            return "after_synthesis"
        elif self.search_results:
            return "after_search"
        elif self.sub_questions:
            return "after_query_decomposition"
        elif self.main_query:
            return "initial"
        else:
            return "empty"

    def update_sub_questions(self, sub_questions: List[str]) -> None:
        """Update the sub-questions list."""
        self.sub_questions = sub_questions

    def update_search_results(
        self, search_results: Dict[str, List[SearchResult]]
    ) -> None:
        """Update the search results."""
        self.search_results = search_results

    def update_synthesized_info(
        self, synthesized_info: Dict[str, SynthesizedInfo]
    ) -> None:
        """Update the synthesized information."""
        self.synthesized_info = synthesized_info

    def update_viewpoint_analysis(
        self, viewpoint_analysis: ViewpointAnalysis
    ) -> None:
        """Update the viewpoint analysis."""
        self.viewpoint_analysis = viewpoint_analysis

    def update_after_reflection(
        self,
        enhanced_info: Dict[str, SynthesizedInfo],
        metadata: ReflectionMetadata,
    ) -> None:
        """Update with reflection results."""
        self.enhanced_info = enhanced_info
        self.reflection_metadata = metadata

    def set_final_report(self, html: str) -> None:
        """Set the final report HTML."""
        self.final_report_html = html


class ApprovalDecision(BaseModel):
    """Approval decision from human reviewer."""

    approved: bool = False
    selected_queries: List[str] = Field(default_factory=list)
    approval_method: str = ""  # "APPROVE_ALL", "SKIP", "SELECT_SPECIFIC"
    reviewer_notes: str = ""
    timestamp: float = Field(default_factory=lambda: time.time())

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class PromptTypeMetrics(BaseModel):
    """Metrics for a specific prompt type."""

    prompt_type: str
    total_cost: float
    input_tokens: int
    output_tokens: int
    call_count: int
    avg_cost_per_call: float
    percentage_of_total_cost: float

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class TracingMetadata(BaseModel):
    """Metadata about token usage, costs, and performance for a pipeline run."""

    # Pipeline information
    pipeline_run_name: str = ""
    pipeline_run_id: str = ""

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    # Cost information
    total_cost: float = 0.0
    cost_breakdown_by_model: Dict[str, float] = Field(default_factory=dict)

    # Performance metrics
    total_latency_seconds: float = 0.0
    formatted_latency: str = ""
    observation_count: int = 0

    # Model usage
    models_used: List[str] = Field(default_factory=list)
    model_token_breakdown: Dict[str, Dict[str, int]] = Field(
        default_factory=dict
    )
    # Format: {"model_name": {"input_tokens": X, "output_tokens": Y, "total_tokens": Z}}

    # Trace information
    trace_id: str = ""
    trace_name: str = ""
    trace_tags: List[str] = Field(default_factory=list)
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Step-by-step breakdown
    step_costs: Dict[str, float] = Field(default_factory=dict)
    step_tokens: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    # Format: {"step_name": {"input_tokens": X, "output_tokens": Y}}

    # Prompt-level metrics
    prompt_metrics: List[PromptTypeMetrics] = Field(
        default_factory=list, description="Cost breakdown by prompt type"
    )

    # Search provider costs
    search_costs: Dict[str, float] = Field(
        default_factory=dict, description="Total costs by search provider"
    )
    search_queries_count: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of queries by search provider",
    )
    search_cost_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed search cost information"
    )

    # Timestamp
    collected_at: float = Field(default_factory=lambda: time.time())

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


# ============================================================================
# New Artifact Classes for ResearchState Refactoring
# ============================================================================


class QueryContext(BaseModel):
    """Immutable context containing the research query and its decomposition.

    This artifact is created once at the beginning of the pipeline and
    remains unchanged throughout execution.
    """

    main_query: str = Field(
        ..., description="The main research question from the user"
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Decomposed sub-questions for parallel processing",
    )
    decomposition_timestamp: float = Field(
        default_factory=lambda: time.time(),
        description="When the query was decomposed",
    )

    model_config = {
        "extra": "ignore",
        "frozen": True,  # Make immutable after creation
        "validate_assignment": True,
    }


class SearchCostDetail(BaseModel):
    """Detailed information about a single search operation."""

    provider: str
    query: str
    cost: float
    timestamp: float
    step: str
    sub_question: Optional[str] = None

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class SearchData(BaseModel):
    """Accumulates search results and cost tracking throughout the pipeline.

    This artifact grows as searches are performed and can be merged
    when parallel searches complete.
    """

    search_results: Dict[str, List[SearchResult]] = Field(
        default_factory=dict,
        description="Map of sub-question to search results",
    )
    search_costs: Dict[str, float] = Field(
        default_factory=dict, description="Total costs by provider"
    )
    search_cost_details: List[SearchCostDetail] = Field(
        default_factory=list, description="Detailed log of each search"
    )
    total_searches: int = Field(
        default=0, description="Total number of searches performed"
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }

    def merge(self, other: "SearchData") -> "SearchData":
        """Merge another SearchData instance into this one."""
        # Merge search results
        for sub_q, results in other.search_results.items():
            if sub_q in self.search_results:
                self.search_results[sub_q].extend(results)
            else:
                self.search_results[sub_q] = results

        # Merge costs
        for provider, cost in other.search_costs.items():
            self.search_costs[provider] = (
                self.search_costs.get(provider, 0.0) + cost
            )

        # Merge cost details
        self.search_cost_details.extend(other.search_cost_details)

        # Update total searches
        self.total_searches += other.total_searches

        return self


class SynthesisData(BaseModel):
    """Contains synthesized information for all sub-questions."""

    synthesized_info: Dict[str, SynthesizedInfo] = Field(
        default_factory=dict,
        description="Synthesized answers for each sub-question",
    )
    enhanced_info: Dict[str, SynthesizedInfo] = Field(
        default_factory=dict,
        description="Enhanced information after reflection",
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }

    def merge(self, other: "SynthesisData") -> "SynthesisData":
        """Merge another SynthesisData instance into this one."""
        self.synthesized_info.update(other.synthesized_info)
        self.enhanced_info.update(other.enhanced_info)
        return self


class AnalysisData(BaseModel):
    """Contains viewpoint analysis and reflection metadata."""

    viewpoint_analysis: Optional[ViewpointAnalysis] = None
    reflection_metadata: Optional[ReflectionMetadata] = None

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }


class FinalReport(BaseModel):
    """Contains the final HTML report."""

    report_html: str = Field(default="", description="The final HTML report")
    generated_at: float = Field(
        default_factory=lambda: time.time(),
        description="Timestamp when report was generated",
    )
    main_query: str = Field(
        default="", description="The original research query"
    )

    model_config = {
        "extra": "ignore",
        "frozen": False,
        "validate_assignment": True,
    }
