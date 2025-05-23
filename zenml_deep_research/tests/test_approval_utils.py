"""Unit tests for approval utility functions."""

from utils.approval_utils import (
    calculate_estimated_cost,
    format_approval_request,
    format_critique_summary,
    format_query_list,
    parse_approval_response,
    summarize_research_progress,
)
from utils.pydantic_models import ResearchState, SynthesizedInfo


def test_parse_approval_responses():
    """Test parsing different approval responses."""
    queries = ["query1", "query2", "query3"]

    # Test approve all
    decision = parse_approval_response("APPROVE ALL", queries)
    assert decision.approved == True
    assert decision.selected_queries == queries
    assert decision.approval_method == "APPROVE_ALL"

    # Test skip
    decision = parse_approval_response(
        "skip", queries
    )  # Test case insensitive
    assert decision.approved == False
    assert decision.selected_queries == []
    assert decision.approval_method == "SKIP"

    # Test selection
    decision = parse_approval_response("SELECT 1,3", queries)
    assert decision.approved == True
    assert decision.selected_queries == ["query1", "query3"]
    assert decision.approval_method == "SELECT_SPECIFIC"

    # Test invalid selection
    decision = parse_approval_response("SELECT invalid", queries)
    assert decision.approved == False
    assert decision.approval_method == "PARSE_ERROR"

    # Test out of range indices
    decision = parse_approval_response("SELECT 1,5,10", queries)
    assert decision.approved == True
    assert decision.selected_queries == ["query1"]  # Only valid indices
    assert decision.approval_method == "SELECT_SPECIFIC"

    # Test unknown response
    decision = parse_approval_response("maybe later", queries)
    assert decision.approved == False
    assert decision.approval_method == "UNKNOWN_RESPONSE"


def test_format_approval_request():
    """Test formatting of approval request messages."""
    message = format_approval_request(
        main_query="Test query",
        progress_summary={
            "completed_count": 5,
            "avg_confidence": 0.75,
            "low_confidence_count": 2,
        },
        critique_points=[
            {"issue": "Missing data", "importance": "high"},
            {"issue": "Minor gap", "importance": "low"},
        ],
        proposed_queries=["query1", "query2"],
    )

    assert "Test query" in message
    assert "5" in message
    assert "0.75" in message
    assert "2 queries" in message
    assert "APPROVE ALL" in message
    assert "SKIP" in message
    assert "SELECT" in message
    assert "Missing data" in message


def test_summarize_research_progress():
    """Test research progress summarization."""
    state = ResearchState(
        main_query="test",
        synthesized_info={
            "q1": SynthesizedInfo(
                synthesized_answer="a1", confidence_level="high"
            ),
            "q2": SynthesizedInfo(
                synthesized_answer="a2", confidence_level="medium"
            ),
            "q3": SynthesizedInfo(
                synthesized_answer="a3", confidence_level="low"
            ),
            "q4": SynthesizedInfo(
                synthesized_answer="a4", confidence_level="low"
            ),
        },
    )

    summary = summarize_research_progress(state)

    assert summary["completed_count"] == 4
    # (1.0 + 0.5 + 0.0 + 0.0) / 4 = 1.5 / 4 = 0.375, rounded to 0.38
    assert summary["avg_confidence"] == 0.38
    assert summary["low_confidence_count"] == 2


def test_format_critique_summary():
    """Test critique summary formatting."""
    # Test with no critiques
    result = format_critique_summary([])
    assert result == "No critical issues identified."

    # Test with few critiques
    critiques = [{"issue": "Issue 1"}, {"issue": "Issue 2"}]
    result = format_critique_summary(critiques)
    assert "- Issue 1" in result
    assert "- Issue 2" in result
    assert "more issues" not in result

    # Test with many critiques
    critiques = [{"issue": f"Issue {i}"} for i in range(5)]
    result = format_critique_summary(critiques)
    assert "- Issue 0" in result
    assert "- Issue 1" in result
    assert "- Issue 2" in result
    assert "- Issue 3" not in result
    assert "... and 2 more issues" in result


def test_format_query_list():
    """Test query list formatting."""
    # Test empty list
    result = format_query_list([])
    assert result == "No queries proposed."

    # Test with queries
    queries = ["Query A", "Query B", "Query C"]
    result = format_query_list(queries)
    assert "1. Query A" in result
    assert "2. Query B" in result
    assert "3. Query C" in result


def test_calculate_estimated_cost():
    """Test cost estimation."""
    assert calculate_estimated_cost([]) == 0.0
    assert calculate_estimated_cost(["q1"]) == 0.01
    assert calculate_estimated_cost(["q1", "q2", "q3"]) == 0.03
    assert calculate_estimated_cost(["q1"] * 10) == 0.10
