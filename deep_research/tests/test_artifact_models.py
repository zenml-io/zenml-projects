"""Tests for the new artifact models."""

import time

import pytest
from utils.pydantic_models import (
    AnalysisData,
    FinalReport,
    QueryContext,
    ReflectionMetadata,
    SearchCostDetail,
    SearchData,
    SearchResult,
    SynthesisData,
    SynthesizedInfo,
    ViewpointAnalysis,
)


class TestQueryContext:
    """Test the QueryContext artifact."""

    def test_query_context_creation(self):
        """Test creating a QueryContext."""
        query = QueryContext(
            main_query="What is quantum computing?",
            sub_questions=["What are qubits?", "How do quantum gates work?"],
        )

        assert query.main_query == "What is quantum computing?"
        assert len(query.sub_questions) == 2
        assert query.decomposition_timestamp > 0

    def test_query_context_immutable(self):
        """Test that QueryContext is immutable."""
        query = QueryContext(main_query="Test query", sub_questions=[])

        # Should raise error when trying to modify
        with pytest.raises(Exception):  # Pydantic will raise validation error
            query.main_query = "Modified query"

    def test_query_context_defaults(self):
        """Test QueryContext with defaults."""
        query = QueryContext(main_query="Test")
        assert query.sub_questions == []
        assert query.decomposition_timestamp > 0


class TestSearchData:
    """Test the SearchData artifact."""

    def test_search_data_creation(self):
        """Test creating SearchData."""
        search_data = SearchData()

        assert search_data.search_results == {}
        assert search_data.search_costs == {}
        assert search_data.search_cost_details == []
        assert search_data.total_searches == 0

    def test_search_data_with_results(self):
        """Test SearchData with actual results."""
        result = SearchResult(
            url="https://example.com",
            content="Test content",
            title="Test Title",
        )

        cost_detail = SearchCostDetail(
            provider="exa",
            query="test query",
            cost=0.01,
            timestamp=time.time(),
            step="process_sub_question",
        )

        search_data = SearchData(
            search_results={"Question 1": [result]},
            search_costs={"exa": 0.01},
            search_cost_details=[cost_detail],
            total_searches=1,
        )

        assert len(search_data.search_results) == 1
        assert search_data.search_costs["exa"] == 0.01
        assert len(search_data.search_cost_details) == 1
        assert search_data.total_searches == 1

    def test_search_data_merge(self):
        """Test merging SearchData instances."""
        # Create first instance
        data1 = SearchData(
            search_results={
                "Q1": [SearchResult(url="url1", content="content1")]
            },
            search_costs={"exa": 0.01},
            total_searches=1,
        )

        # Create second instance
        data2 = SearchData(
            search_results={
                "Q1": [SearchResult(url="url2", content="content2")],
                "Q2": [SearchResult(url="url3", content="content3")],
            },
            search_costs={"exa": 0.02, "tavily": 0.01},
            total_searches=2,
        )

        # Merge
        data1.merge(data2)

        # Check results
        assert len(data1.search_results["Q1"]) == 2  # Merged Q1 results
        assert "Q2" in data1.search_results  # Added Q2
        assert data1.search_costs["exa"] == 0.03  # Combined costs
        assert data1.search_costs["tavily"] == 0.01  # New provider
        assert data1.total_searches == 3


class TestSynthesisData:
    """Test the SynthesisData artifact."""

    def test_synthesis_data_creation(self):
        """Test creating SynthesisData."""
        synthesis = SynthesisData()

        assert synthesis.synthesized_info == {}
        assert synthesis.enhanced_info == {}

    def test_synthesis_data_with_info(self):
        """Test SynthesisData with synthesized info."""
        synth_info = SynthesizedInfo(
            synthesized_answer="Test answer",
            key_sources=["source1", "source2"],
            confidence_level="high",
        )

        synthesis = SynthesisData(synthesized_info={"Q1": synth_info})

        assert "Q1" in synthesis.synthesized_info
        assert synthesis.synthesized_info["Q1"].confidence_level == "high"

    def test_synthesis_data_merge(self):
        """Test merging SynthesisData instances."""
        info1 = SynthesizedInfo(synthesized_answer="Answer 1")
        info2 = SynthesizedInfo(synthesized_answer="Answer 2")

        data1 = SynthesisData(synthesized_info={"Q1": info1})
        data2 = SynthesisData(synthesized_info={"Q2": info2})

        data1.merge(data2)

        assert "Q1" in data1.synthesized_info
        assert "Q2" in data1.synthesized_info


class TestAnalysisData:
    """Test the AnalysisData artifact."""

    def test_analysis_data_creation(self):
        """Test creating AnalysisData."""
        analysis = AnalysisData()

        assert analysis.viewpoint_analysis is None
        assert analysis.reflection_metadata is None

    def test_analysis_data_with_viewpoint(self):
        """Test AnalysisData with viewpoint analysis."""
        viewpoint = ViewpointAnalysis(
            main_points_of_agreement=["Point 1", "Point 2"],
            perspective_gaps="Some gaps",
        )

        analysis = AnalysisData(viewpoint_analysis=viewpoint)

        assert analysis.viewpoint_analysis is not None
        assert len(analysis.viewpoint_analysis.main_points_of_agreement) == 2

    def test_analysis_data_with_reflection(self):
        """Test AnalysisData with reflection metadata."""
        reflection = ReflectionMetadata(
            critique_summary=["Critique 1"], improvements_made=3.0
        )

        analysis = AnalysisData(reflection_metadata=reflection)

        assert analysis.reflection_metadata is not None
        assert analysis.reflection_metadata.improvements_made == 3.0


class TestFinalReport:
    """Test the FinalReport artifact."""

    def test_final_report_creation(self):
        """Test creating FinalReport."""
        report = FinalReport()

        assert report.report_html == ""
        assert report.generated_at > 0
        assert report.main_query == ""

    def test_final_report_with_content(self):
        """Test FinalReport with HTML content."""
        html = "<html><body>Test Report</body></html>"
        report = FinalReport(report_html=html, main_query="What is AI?")

        assert report.report_html == html
        assert report.main_query == "What is AI?"
        assert report.generated_at > 0
