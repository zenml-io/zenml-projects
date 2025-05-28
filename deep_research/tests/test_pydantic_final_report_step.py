"""Tests for the Pydantic-based final report step.

This module contains tests for the Pydantic-based implementation of
final_report_step, which uses the new Pydantic models and materializers.
"""

from typing import Dict, List

import pytest
from steps.pydantic_final_report_step import pydantic_final_report_step
from utils.pydantic_models import (
    AnalysisData,
    FinalReport,
    Prompt,
    QueryContext,
    ReflectionMetadata,
    SearchData,
    SearchResult,
    SynthesisData,
    SynthesizedInfo,
    ViewpointAnalysis,
    ViewpointTension,
)
from zenml.types import HTMLString


@pytest.fixture
def sample_artifacts():
    """Create sample artifacts for testing."""
    # Create QueryContext
    query_context = QueryContext(
        main_query="What are the impacts of climate change?",
        sub_questions=["Economic impacts", "Environmental impacts"],
    )

    # Create SearchData
    search_results: Dict[str, List[SearchResult]] = {
        "Economic impacts": [
            SearchResult(
                url="https://example.com/economy",
                title="Economic Impacts of Climate Change",
                snippet="Overview of economic impacts",
                content="Detailed content about economic impacts of climate change",
            )
        ],
        "Environmental impacts": [
            SearchResult(
                url="https://example.com/environment",
                title="Environmental Impacts",
                snippet="Environmental impact overview",
                content="Content about environmental impacts",
            )
        ],
    }
    search_data = SearchData(search_results=search_results)

    # Create SynthesisData
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
    synthesis_data = SynthesisData(
        synthesized_info=synthesized_info,
        enhanced_info=synthesized_info,  # Same as synthesized for this test
    )

    # Create AnalysisData
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

    reflection_metadata = ReflectionMetadata(
        critique_summary=["Need more sources for economic impacts"],
        additional_questions_identified=[
            "How will climate change affect different regions?"
        ],
        searches_performed=[
            "economic impacts of climate change",
            "regional climate impacts",
        ],
        improvements_made=2.0,
    )

    analysis_data = AnalysisData(
        viewpoint_analysis=viewpoint_analysis,
        reflection_metadata=reflection_metadata,
    )

    # Create prompts
    conclusion_prompt = Prompt(
        name="conclusion_generation",
        content="Generate a conclusion based on the research findings.",
    )
    executive_summary_prompt = Prompt(
        name="executive_summary", content="Generate an executive summary."
    )
    introduction_prompt = Prompt(
        name="introduction", content="Generate an introduction."
    )

    return {
        "query_context": query_context,
        "search_data": search_data,
        "synthesis_data": synthesis_data,
        "analysis_data": analysis_data,
        "conclusion_generation_prompt": conclusion_prompt,
        "executive_summary_prompt": executive_summary_prompt,
        "introduction_prompt": introduction_prompt,
    }


def test_pydantic_final_report_step_returns_tuple():
    """Test that the step returns a tuple with FinalReport and HTML."""
    # Create simple artifacts
    query_context = QueryContext(
        main_query="What is climate change?",
        sub_questions=["What causes climate change?"],
    )
    search_data = SearchData()
    synthesis_data = SynthesisData(
        synthesized_info={
            "What causes climate change?": SynthesizedInfo(
                synthesized_answer="Climate change is caused by greenhouse gases.",
                confidence_level="high",
                key_sources=["https://example.com/causes"],
            )
        }
    )
    analysis_data = AnalysisData()

    # Create prompts
    conclusion_prompt = Prompt(
        name="conclusion_generation", content="Generate a conclusion."
    )
    executive_summary_prompt = Prompt(
        name="executive_summary", content="Generate summary."
    )
    introduction_prompt = Prompt(
        name="introduction", content="Generate intro."
    )

    # Run the step
    result = pydantic_final_report_step(
        query_context=query_context,
        search_data=search_data,
        synthesis_data=synthesis_data,
        analysis_data=analysis_data,
        conclusion_generation_prompt=conclusion_prompt,
        executive_summary_prompt=executive_summary_prompt,
        introduction_prompt=introduction_prompt,
    )

    # Assert that result is a tuple with 2 elements
    assert isinstance(result, tuple)
    assert len(result) == 2

    # Assert first element is FinalReport
    assert isinstance(result[0], FinalReport)

    # Assert second element is HTMLString
    assert isinstance(result[1], HTMLString)


def test_pydantic_final_report_step_with_complex_artifacts(sample_artifacts):
    """Test that the step handles complex artifacts properly."""
    # Run the step with complex artifacts
    result = pydantic_final_report_step(
        query_context=sample_artifacts["query_context"],
        search_data=sample_artifacts["search_data"],
        synthesis_data=sample_artifacts["synthesis_data"],
        analysis_data=sample_artifacts["analysis_data"],
        conclusion_generation_prompt=sample_artifacts[
            "conclusion_generation_prompt"
        ],
        executive_summary_prompt=sample_artifacts["executive_summary_prompt"],
        introduction_prompt=sample_artifacts["introduction_prompt"],
    )

    # Unpack the results
    final_report, html_report = result

    # Assert FinalReport contains expected data
    assert final_report.main_query == "What are the impacts of climate change?"
    assert len(final_report.sub_questions) == 2
    assert final_report.report_html != ""

    # Assert HTML report contains key elements
    html_str = str(html_report)
    assert "Economic impacts" in html_str
    assert "Environmental impacts" in html_str
    assert "Viewpoint Analysis" in html_str
    assert "Progressive" in html_str
    assert "Conservative" in html_str


def test_pydantic_final_report_step_creates_report():
    """Test that the step properly creates a final report."""
    # Create artifacts
    query_context = QueryContext(
        main_query="What is climate change?",
        sub_questions=["What causes climate change?"],
    )
    search_data = SearchData()
    synthesis_data = SynthesisData(
        synthesized_info={
            "What causes climate change?": SynthesizedInfo(
                synthesized_answer="Climate change is caused by greenhouse gases.",
                confidence_level="high",
                key_sources=["https://example.com/causes"],
            )
        }
    )
    analysis_data = AnalysisData()

    # Create prompts
    conclusion_prompt = Prompt(
        name="conclusion_generation", content="Generate a conclusion."
    )
    executive_summary_prompt = Prompt(
        name="executive_summary", content="Generate summary."
    )
    introduction_prompt = Prompt(
        name="introduction", content="Generate intro."
    )

    # Run the step
    final_report, html_report = pydantic_final_report_step(
        query_context=query_context,
        search_data=search_data,
        synthesis_data=synthesis_data,
        analysis_data=analysis_data,
        conclusion_generation_prompt=conclusion_prompt,
        executive_summary_prompt=executive_summary_prompt,
        introduction_prompt=introduction_prompt,
    )

    # Verify FinalReport was created with content
    assert final_report.report_html != ""
    assert "climate change" in final_report.report_html.lower()

    # Verify HTML report was created
    assert str(html_report) != ""
    assert "climate change" in str(html_report).lower()
