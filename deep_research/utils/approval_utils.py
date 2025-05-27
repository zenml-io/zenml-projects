"""Utility functions for the human approval process."""

from typing import Any, Dict, List

from utils.pydantic_models import ApprovalDecision


def summarize_research_progress(state) -> Dict[str, Any]:
    """Summarize the current research progress."""
    completed_count = len(state.synthesized_info)
    confidence_levels = [
        info.confidence_level for info in state.synthesized_info.values()
    ]

    # Calculate average confidence (high=1.0, medium=0.5, low=0.0)
    confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    avg_confidence = sum(
        confidence_map.get(c, 0.5) for c in confidence_levels
    ) / max(len(confidence_levels), 1)

    low_confidence_count = sum(1 for c in confidence_levels if c == "low")

    return {
        "completed_count": completed_count,
        "avg_confidence": round(avg_confidence, 2),
        "low_confidence_count": low_confidence_count,
    }


def format_critique_summary(critique_points: List[Dict[str, Any]]) -> str:
    """Format critique points for display."""
    if not critique_points:
        return "No critical issues identified."

    formatted = []
    for point in critique_points[:3]:  # Show top 3
        issue = point.get("issue", "Unknown issue")
        formatted.append(f"- {issue}")

    if len(critique_points) > 3:
        formatted.append(f"- ... and {len(critique_points) - 3} more issues")

    return "\n".join(formatted)


def format_query_list(queries: List[str]) -> str:
    """Format query list for display."""
    if not queries:
        return "No queries proposed."

    formatted = []
    for i, query in enumerate(queries, 1):
        formatted.append(f"{i}. {query}")

    return "\n".join(formatted)


def calculate_estimated_cost(queries: List[str]) -> float:
    """Calculate estimated cost for additional queries."""
    # Rough estimate: ~$0.01 per query (including search API + LLM costs)
    return round(len(queries) * 0.01, 2)


def format_approval_request(
    main_query: str,
    progress_summary: Dict[str, Any],
    critique_points: List[Dict[str, Any]],
    proposed_queries: List[str],
    timeout: int = 3600,
) -> str:
    """Format the approval request message."""

    # High-priority critiques
    high_priority = [
        c for c in critique_points if c.get("importance") == "high"
    ]

    message = f"""📊 **Research Progress Update**

**Main Query:** {main_query}

**Current Status:**
- Sub-questions analyzed: {progress_summary["completed_count"]}
- Average confidence: {progress_summary["avg_confidence"]}
- Low confidence areas: {progress_summary["low_confidence_count"]}

**Key Issues Identified:**
{format_critique_summary(high_priority or critique_points)}

**Proposed Additional Research** ({len(proposed_queries)} queries):
{format_query_list(proposed_queries)}

**Estimated Additional Time:** ~{len(proposed_queries) * 2} minutes
**Estimated Additional Cost:** ~${calculate_estimated_cost(proposed_queries)}

**Response Options:**
- Reply with `approve`, `yes`, `ok`, or `LGTM` to proceed with all queries
- Reply with `reject`, `no`, `skip`, or `decline` to finish with current findings

**Timeout:** Response required within {timeout // 60} minutes"""

    return message


def parse_approval_response(
    response: str, proposed_queries: List[str]
) -> ApprovalDecision:
    """Parse the approval response from user."""

    response_upper = response.strip().upper()

    if response_upper == "APPROVE ALL":
        return ApprovalDecision(
            approved=True,
            selected_queries=proposed_queries,
            approval_method="APPROVE_ALL",
            reviewer_notes=response,
        )

    elif response_upper == "SKIP":
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="SKIP",
            reviewer_notes=response,
        )

    elif response_upper.startswith("SELECT"):
        # Parse selection like "SELECT 1,3,5"
        try:
            # Extract the part after "SELECT"
            selection_part = response_upper[6:].strip()
            indices = [int(x.strip()) - 1 for x in selection_part.split(",")]
            selected = [
                proposed_queries[i]
                for i in indices
                if 0 <= i < len(proposed_queries)
            ]
            return ApprovalDecision(
                approved=True,
                selected_queries=selected,
                approval_method="SELECT_SPECIFIC",
                reviewer_notes=response,
            )
        except Exception as e:
            return ApprovalDecision(
                approved=False,
                selected_queries=[],
                approval_method="PARSE_ERROR",
                reviewer_notes=f"Failed to parse: {response} - {str(e)}",
            )

    else:
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="UNKNOWN_RESPONSE",
            reviewer_notes=f"Unknown response: {response}",
        )
