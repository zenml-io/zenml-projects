"""Tests for the Pydantic-based final report step.

This module contains tests for the Pydantic-based implementation of
final_report_step, which uses the new Pydantic models and materializers.
"""

from typing import Dict, List

import pytest
from steps.pydantic_final_report_step import pydantic_final_report_step
from utils.pydantic_models import (
    ReflectionMetadata,
    ResearchState,
    SearchResult,
    SynthesizedInfo,
    ViewpointAnalysis,
    ViewpointTension,
)
from zenml.types import HTMLString


@pytest.fixture
def sample_research_state() -> ResearchState:
    """Create a sample research state for testing."""
    # Create a basic research state
    state = ResearchState(main_query="What are the impacts of climate change?")

    # Add sub-questions
    state.update_sub_questions(["Economic impacts", "Environmental impacts"])

    # Add search results
    search_results: Dict[str, List[SearchResult]] = {
        "Economic impacts": [
            SearchResult(
                url="https://example.com/economy",
                title="Economic Impacts of Climate Change",
                snippet="Overview of economic impacts",
                content="Detailed content about economic impacts of climate change",
            )
        ]
    }
    state.update_search_results(search_results)

    # Add synthesized info
    synthesized_info: Dict[str, SynthesizedInfo] = {
        "Economic impacts": SynthesizedInfo(
            synthesized_answer="Climate change will have significant economic impacts...",
            key_sources=["https://example.com/economy"],
            confidence_level="high",
        ),
        "Environmental impacts": SynthesizedInfo(
            synthesized_answer="Environmental impacts include rising sea levels...",
            key_sources=["https://example.com/environment"],
            confidence_level="high",
        ),
    }
    state.update_synthesized_info(synthesized_info)

    # Add enhanced info (same as synthesized for this test)
    state.enhanced_info = state.synthesized_info

    # Add viewpoint analysis
    viewpoint_analysis = ViewpointAnalysis(
        main_points_of_agreement=[
            "Climate change is happening",
            "Action is needed",
        ],
        areas_of_tension=[
            ViewpointTension(
                topic="Economic policy",
                viewpoints={
                    "Progressive": "Support carbon taxes and regulations",
                    "Conservative": "Prefer market-based solutions",
                },
            )
        ],
        perspective_gaps="Indigenous perspectives are underrepresented",
        integrative_insights="A balanced approach combining regulations and market incentives may be most effective",
    )
    state.update_viewpoint_analysis(viewpoint_analysis)

    # Add reflection metadata
    reflection_metadata = ReflectionMetadata(
        critique_summary=["Need more sources for economic impacts"],
        additional_questions_identified=[
            "How will climate change affect different regions?"
        ],
        searches_performed=[
            "economic impacts of climate change",
            "regional climate impacts",
        ],
        improvements_made=2,
    )
    state.reflection_metadata = reflection_metadata

    return state


def test_pydantic_final_report_step_returns_tuple():
    """Test that the step returns a tuple with state and HTML."""
    # Create a simple state
    state = ResearchState(main_query="What is climate change?")
    state.update_sub_questions(["What causes climate change?"])

    # Run the step
    result = pydantic_final_report_step(state=state)

    # Assert that result is a tuple with 2 elements
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Assert first element is ResearchState
    assert isinstance(result[0], ResearchState)

    # Assert second element is HTMLString
    assert isinstance(result[1], HTMLString)


def test_pydantic_final_report_step_with_complex_state(sample_research_state):
    """Test that the step handles a complex state properly."""
    # Run the step with a complex state
    result = pydantic_final_report_step(state=sample_research_state)

    # Unpack the results
    updated_state, html_report = result

    # Assert state contains final report HTML
    assert updated_state.final_report_html != ""

    # Assert HTML report contains key elements
    html_str = str(html_report)
    assert "Economic impacts" in html_str
    assert "Environmental impacts" in html_str
    assert "Viewpoint Analysis" in html_str
    assert "Progressive" in html_str
    assert "Conservative" in html_str


def test_pydantic_final_report_step_updates_state():
    """Test that the step properly updates the state."""
    # Create an initial state without a final report
    state = ResearchState(
        main_query="What is climate change?",
        sub_questions=["What causes climate change?"],
        synthesized_info={
            "What causes climate change?": SynthesizedInfo(
                synthesized_answer="Climate change is caused by greenhouse gases.",
                confidence_level="high",
            )
        },
        enhanced_info={
            "What causes climate change?": SynthesizedInfo(
                synthesized_answer="Climate change is caused by greenhouse gases.",
                confidence_level="high",
            )
        },
    )

    # Verify initial state has no report
    assert state.final_report_html == ""

    # Run the step
    updated_state, _ = pydantic_final_report_step(state=state)

    # Verify state was updated with a report
    assert updated_state.final_report_html != ""
    assert "climate change" in updated_state.final_report_html.lower()
