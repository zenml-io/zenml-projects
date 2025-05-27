"""Tests for Pydantic model implementations.

This module contains tests for the Pydantic models that validate:
1. Basic model instantiation
2. Default values
3. Serialization and deserialization
4. Method functionality
"""

import json
from typing import Dict, List

from utils.pydantic_models import (
    ReflectionMetadata,
    ResearchState,
    SearchResult,
    SynthesizedInfo,
    ViewpointAnalysis,
    ViewpointTension,
)


def test_search_result_creation():
    """Test creating a SearchResult model."""
    # Create with defaults
    result = SearchResult()
    assert result.url == ""
    assert result.content == ""
    assert result.title == ""
    assert result.snippet == ""

    # Create with values
    result = SearchResult(
        url="https://example.com",
        content="Example content",
        title="Example Title",
        snippet="This is a snippet",
    )
    assert result.url == "https://example.com"
    assert result.content == "Example content"
    assert result.title == "Example Title"
    assert result.snippet == "This is a snippet"


def test_search_result_serialization():
    """Test serializing and deserializing a SearchResult."""
    result = SearchResult(
        url="https://example.com",
        content="Example content",
        title="Example Title",
        snippet="This is a snippet",
    )

    # Serialize to dict
    result_dict = result.model_dump()
    assert result_dict["url"] == "https://example.com"
    assert result_dict["content"] == "Example content"

    # Serialize to JSON
    result_json = result.model_dump_json()
    result_dict_from_json = json.loads(result_json)
    assert result_dict_from_json["url"] == "https://example.com"

    # Deserialize from dict
    new_result = SearchResult.model_validate(result_dict)
    assert new_result.url == "https://example.com"
    assert new_result.content == "Example content"

    # Deserialize from JSON
    new_result_from_json = SearchResult.model_validate_json(result_json)
    assert new_result_from_json.url == "https://example.com"


def test_viewpoint_tension_model():
    """Test the ViewpointTension model."""
    # Empty model
    tension = ViewpointTension()
    assert tension.topic == ""
    assert tension.viewpoints == {}

    # With data
    tension = ViewpointTension(
        topic="Climate Change Impacts",
        viewpoints={
            "Economic": "Focuses on financial costs and benefits",
            "Environmental": "Emphasizes ecosystem impacts",
        },
    )
    assert tension.topic == "Climate Change Impacts"
    assert len(tension.viewpoints) == 2
    assert "Economic" in tension.viewpoints

    # Serialization
    tension_dict = tension.model_dump()
    assert tension_dict["topic"] == "Climate Change Impacts"
    assert len(tension_dict["viewpoints"]) == 2

    # Deserialization
    new_tension = ViewpointTension.model_validate(tension_dict)
    assert new_tension.topic == tension.topic
    assert new_tension.viewpoints == tension.viewpoints


def test_synthesized_info_model():
    """Test the SynthesizedInfo model."""
    # Default values
    info = SynthesizedInfo()
    assert info.synthesized_answer == ""
    assert info.key_sources == []
    assert info.confidence_level == "medium"
    assert info.information_gaps == ""
    assert info.improvements == []

    # With values
    info = SynthesizedInfo(
        synthesized_answer="This is a synthesized answer",
        key_sources=["https://source1.com", "https://source2.com"],
        confidence_level="high",
        information_gaps="Missing some context",
        improvements=["Add more detail", "Check more sources"],
    )
    assert info.synthesized_answer == "This is a synthesized answer"
    assert len(info.key_sources) == 2
    assert info.confidence_level == "high"

    # Serialization and deserialization
    info_dict = info.model_dump()
    new_info = SynthesizedInfo.model_validate(info_dict)
    assert new_info.synthesized_answer == info.synthesized_answer
    assert new_info.key_sources == info.key_sources


def test_viewpoint_analysis_model():
    """Test the ViewpointAnalysis model."""
    # Create tensions for the analysis
    tension1 = ViewpointTension(
        topic="Economic Impact",
        viewpoints={
            "Positive": "Creates jobs",
            "Negative": "Increases inequality",
        },
    )
    tension2 = ViewpointTension(
        topic="Environmental Impact",
        viewpoints={
            "Positive": "Reduces emissions",
            "Negative": "Land use changes",
        },
    )

    # Create the analysis
    analysis = ViewpointAnalysis(
        main_points_of_agreement=[
            "Need for action",
            "Technological innovation",
        ],
        areas_of_tension=[tension1, tension2],
        perspective_gaps="Missing indigenous perspectives",
        integrative_insights="Combined economic and environmental approach needed",
    )

    assert len(analysis.main_points_of_agreement) == 2
    assert len(analysis.areas_of_tension) == 2
    assert analysis.areas_of_tension[0].topic == "Economic Impact"

    # Test serialization
    analysis_dict = analysis.model_dump()
    assert len(analysis_dict["areas_of_tension"]) == 2
    assert analysis_dict["areas_of_tension"][0]["topic"] == "Economic Impact"

    # Test deserialization
    new_analysis = ViewpointAnalysis.model_validate(analysis_dict)
    assert len(new_analysis.areas_of_tension) == 2
    assert new_analysis.areas_of_tension[0].topic == "Economic Impact"
    assert new_analysis.perspective_gaps == analysis.perspective_gaps


def test_reflection_metadata_model():
    """Test the ReflectionMetadata model."""
    metadata = ReflectionMetadata(
        critique_summary=["Need more sources", "Missing detailed analysis"],
        additional_questions_identified=["What about future trends?"],
        searches_performed=["future climate trends", "economic impacts"],
        improvements_made=3,
        error=None,
    )

    assert len(metadata.critique_summary) == 2
    assert len(metadata.additional_questions_identified) == 1
    assert metadata.improvements_made == 3
    assert metadata.error is None

    # Serialization
    metadata_dict = metadata.model_dump()
    assert len(metadata_dict["critique_summary"]) == 2
    assert metadata_dict["improvements_made"] == 3

    # Deserialization
    new_metadata = ReflectionMetadata.model_validate(metadata_dict)
    assert new_metadata.improvements_made == metadata.improvements_made
    assert new_metadata.critique_summary == metadata.critique_summary


def test_research_state_model():
    """Test the main ResearchState model."""
    # Create with defaults
    state = ResearchState()
    assert state.main_query == ""
    assert state.sub_questions == []
    assert state.search_results == {}
    assert state.get_current_stage() == "empty"

    # Set main query
    state.main_query = "What are the impacts of climate change?"
    assert state.get_current_stage() == "initial"

    # Test update methods
    state.update_sub_questions(
        ["What are economic impacts?", "What are environmental impacts?"]
    )
    assert len(state.sub_questions) == 2
    assert state.get_current_stage() == "after_query_decomposition"

    # Add search results
    search_results: Dict[str, List[SearchResult]] = {
        "What are economic impacts?": [
            SearchResult(
                url="https://example.com/economy",
                title="Economic Impacts",
                snippet="Overview of economic impacts",
                content="Detailed content about economic impacts",
            )
        ]
    }
    state.update_search_results(search_results)
    assert state.get_current_stage() == "after_search"
    assert len(state.search_results["What are economic impacts?"]) == 1

    # Add synthesized info
    synthesized_info: Dict[str, SynthesizedInfo] = {
        "What are economic impacts?": SynthesizedInfo(
            synthesized_answer="Economic impacts include job losses and growth opportunities",
            key_sources=["https://example.com/economy"],
            confidence_level="high",
        )
    }
    state.update_synthesized_info(synthesized_info)
    assert state.get_current_stage() == "after_synthesis"

    # Add viewpoint analysis
    analysis = ViewpointAnalysis(
        main_points_of_agreement=["Economic changes are happening"],
        areas_of_tension=[
            ViewpointTension(
                topic="Job impacts",
                viewpoints={
                    "Positive": "New green jobs",
                    "Negative": "Fossil fuel job losses",
                },
            )
        ],
    )
    state.update_viewpoint_analysis(analysis)
    assert state.get_current_stage() == "after_viewpoint_analysis"

    # Add reflection results
    enhanced_info = {
        "What are economic impacts?": SynthesizedInfo(
            synthesized_answer="Enhanced answer with more details",
            key_sources=[
                "https://example.com/economy",
                "https://example.com/new-source",
            ],
            confidence_level="high",
            improvements=["Added more context", "Added more sources"],
        )
    }
    metadata = ReflectionMetadata(
        critique_summary=["Needed more sources"],
        improvements_made=2,
    )
    state.update_after_reflection(enhanced_info, metadata)
    assert state.get_current_stage() == "after_reflection"

    # Set final report
    state.set_final_report("<html>Final report content</html>")
    assert state.get_current_stage() == "final_report"
    assert state.final_report_html == "<html>Final report content</html>"

    # Test serialization and deserialization
    state_dict = state.model_dump()
    new_state = ResearchState.model_validate(state_dict)

    # Verify key properties were preserved
    assert new_state.main_query == state.main_query
    assert len(new_state.sub_questions) == len(state.sub_questions)
    assert new_state.get_current_stage() == state.get_current_stage()
    assert new_state.viewpoint_analysis is not None
    assert len(new_state.viewpoint_analysis.areas_of_tension) == 1
    assert (
        new_state.viewpoint_analysis.areas_of_tension[0].topic == "Job impacts"
    )
    assert new_state.final_report_html == state.final_report_html
