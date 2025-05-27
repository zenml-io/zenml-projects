"""Tests for Pydantic-based materializer.

This module contains tests for the Pydantic-based implementation of
ResearchStateMaterializer, verifying that it correctly serializes and
visualizes ResearchState objects.
"""

import os
import tempfile
from typing import Dict, List

import pytest
from materializers.pydantic_materializer import ResearchStateMaterializer
from utils.pydantic_models import (
    ResearchState,
    SearchResult,
    SynthesizedInfo,
    ViewpointAnalysis,
    ViewpointTension,
)


@pytest.fixture
def sample_state() -> ResearchState:
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
        )
    }
    state.update_synthesized_info(synthesized_info)

    return state


def test_materializer_initialization():
    """Test that the materializer can be initialized."""
    # Create a temporary directory for artifact storage
    with tempfile.TemporaryDirectory() as tmpdirname:
        materializer = ResearchStateMaterializer(uri=tmpdirname)
        assert materializer is not None


def test_materializer_save_and_load(sample_state: ResearchState):
    """Test saving and loading a state using the materializer."""
    # Create a temporary directory for artifact storage
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize materializer with temporary artifact URI
        materializer = ResearchStateMaterializer(uri=tmpdirname)

        # Save the state
        materializer.save(sample_state)

        # Load the state
        loaded_state = materializer.load(ResearchState)

        # Verify that the loaded state matches the original
        assert loaded_state.main_query == sample_state.main_query
        assert loaded_state.sub_questions == sample_state.sub_questions
        assert len(loaded_state.search_results) == len(
            sample_state.search_results
        )
        assert (
            loaded_state.get_current_stage()
            == sample_state.get_current_stage()
        )

        # Check that key fields were preserved
        question = "Economic impacts"
        assert (
            loaded_state.synthesized_info[question].synthesized_answer
            == sample_state.synthesized_info[question].synthesized_answer
        )
        assert (
            loaded_state.synthesized_info[question].confidence_level
            == sample_state.synthesized_info[question].confidence_level
        )


def test_materializer_save_visualizations(sample_state: ResearchState):
    """Test generating and saving visualizations."""
    # Create a temporary directory for artifact storage
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize materializer with temporary artifact URI
        materializer = ResearchStateMaterializer(uri=tmpdirname)

        # Generate and save visualizations
        viz_paths = materializer.save_visualizations(sample_state)

        # Verify visualization file exists
        html_path = list(viz_paths.keys())[0]
        assert os.path.exists(html_path)

        # Verify the file has content
        with open(html_path, "r") as f:
            content = f.read()
            # Check for expected elements in the HTML
            assert "Research State" in content
            assert sample_state.main_query in content
            assert "Economic impacts" in content


def test_html_generation_stages(sample_state: ResearchState):
    """Test that HTML visualization reflects the correct research stage."""
    # Create the materializer
    with tempfile.TemporaryDirectory() as tmpdirname:
        materializer = ResearchStateMaterializer(uri=tmpdirname)

        # Generate visualization at initial state
        html = materializer._generate_visualization_html(sample_state)
        # Verify stage by checking for expected elements in the HTML
        assert (
            "Synthesized Information" in html
        )  # Should show synthesized info

        # Add viewpoint analysis
        state_with_viewpoints = sample_state.model_copy(deep=True)
        viewpoint_analysis = ViewpointAnalysis(
            main_points_of_agreement=["There will be economic impacts"],
            areas_of_tension=[
                ViewpointTension(
                    topic="Job impacts",
                    viewpoints={
                        "Positive": "New green jobs",
                        "Negative": "Job losses",
                    },
                )
            ],
        )
        state_with_viewpoints.update_viewpoint_analysis(viewpoint_analysis)
        html = materializer._generate_visualization_html(state_with_viewpoints)
        assert "Viewpoint Analysis" in html
        assert "Points of Agreement" in html

        # Add final report
        state_with_report = state_with_viewpoints.model_copy(deep=True)
        state_with_report.set_final_report("<html>Final report content</html>")
        html = materializer._generate_visualization_html(state_with_report)
        assert "Final Report" in html
