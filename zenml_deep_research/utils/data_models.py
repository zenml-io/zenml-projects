from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Search:
    """Represents a search result from Tavily."""

    url: str = ""
    content: str = ""


@dataclass
class Research:
    """Container for search history and research progress for a paragraph."""

    search_history: List[Search] = field(default_factory=list)
    latest_summary: str = ""
    reflection_iteration: int = 0


@dataclass
class Paragraph:
    """Represents a paragraph in the research report."""

    title: str = ""
    content: str = ""
    research: Research = field(default_factory=Research)


@dataclass
class State:
    """The overall state of the research process."""

    report_title: str = ""
    query: str = ""  # Added to keep track of the original query
    paragraphs: List[Paragraph] = field(default_factory=list)


@dataclass
class SearchResult:
    """Represents a search result for a sub-question."""

    url: str = ""
    content: str = ""
    title: str = ""
    snippet: str = ""


@dataclass
class SynthesizedInfo:
    """Represents synthesized information for a sub-question."""

    synthesized_answer: str = ""
    key_sources: List[str] = field(default_factory=list)
    confidence_level: str = "medium"  # high, medium, low
    information_gaps: str = ""
    improvements: List[str] = field(default_factory=list)


@dataclass
class ViewpointTension:
    """Represents a tension between different viewpoints on a topic."""

    topic: str = ""
    viewpoints: Dict[str, str] = field(default_factory=dict)


@dataclass
class ViewpointAnalysis:
    """Represents the analysis of different viewpoints on the research topic."""

    main_points_of_agreement: List[str] = field(default_factory=list)
    areas_of_tension: List[ViewpointTension] = field(default_factory=list)
    perspective_gaps: str = ""
    integrative_insights: str = ""


@dataclass
class CritiqueItem:
    """Represents a single critique item of the research."""

    area: str = ""
    issue: str = ""
    importance: str = "medium"  # high, medium, low


@dataclass
class ReflectionMetadata:
    """Metadata about the reflection process."""

    critique_summary: List[str] = field(default_factory=list)
    additional_questions_identified: List[str] = field(default_factory=list)
    searches_performed: List[str] = field(default_factory=list)
    improvements_made: int = 0
    error: Optional[str] = None


@dataclass
class ResearchState:
    """Comprehensive state object for the enhanced research pipeline."""

    # Initial query information
    main_query: str = ""
    sub_questions: List[str] = field(default_factory=list)

    # Information gathering results
    search_results: Dict[str, List[SearchResult]] = field(default_factory=dict)

    # Synthesized information
    synthesized_info: Dict[str, SynthesizedInfo] = field(default_factory=dict)

    # Viewpoint analysis
    viewpoint_analysis: Optional[ViewpointAnalysis] = None

    # Reflection results
    enhanced_info: Dict[str, SynthesizedInfo] = field(default_factory=dict)
    reflection_metadata: Optional[ReflectionMetadata] = None

    # Final report
    final_report_html: str = ""

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
