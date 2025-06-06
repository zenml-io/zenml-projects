import logging
import time
from typing import Annotated, List

from materializers.approval_decision_materializer import (
    ApprovalDecisionMaterializer,
)
from utils.approval_utils import (
    format_approval_request,
)
from utils.pydantic_models import (
    AnalysisData,
    ApprovalDecision,
    QueryContext,
    SynthesisData,
)
from zenml import log_metadata, step
from zenml.client import Client

logger = logging.getLogger(__name__)


def summarize_research_progress_from_artifacts(
    synthesis_data: SynthesisData, analysis_data: AnalysisData
) -> dict:
    """Summarize research progress from the new artifact structure."""
    completed_count = len(synthesis_data.synthesized_info)

    # Calculate confidence levels from synthesis data
    confidence_levels = []
    for info in synthesis_data.synthesized_info.values():
        confidence_levels.append(info.confidence_level)

    # Calculate average confidence (high=1.0, medium=0.5, low=0.0)
    confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
    avg_confidence = sum(
        confidence_map.get(c.lower(), 0.5) for c in confidence_levels
    ) / max(len(confidence_levels), 1)

    low_confidence_count = sum(
        1 for c in confidence_levels if c.lower() == "low"
    )

    return {
        "completed_count": completed_count,
        "avg_confidence": round(avg_confidence, 2),
        "low_confidence_count": low_confidence_count,
    }


@step(
    enable_cache=False,
    output_materializers={"approval_decision": ApprovalDecisionMaterializer},
)  # Never cache approval decisions
def get_research_approval_step(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    recommended_queries: List[str],
    require_approval: bool = True,
    alerter_type: str = "slack",
    timeout: int = 3600,
    max_queries: int = 2,
) -> Annotated[ApprovalDecision, "approval_decision"]:
    """
    Get human approval for additional research queries.

    Always returns an ApprovalDecision object. If require_approval is False,
    automatically approves all queries.

    Args:
        query_context: Context containing the main query and sub-questions
        synthesis_data: Synthesized information from research
        analysis_data: Analysis including viewpoints and critique
        recommended_queries: List of recommended additional queries
        require_approval: Whether to require human approval
        alerter_type: Type of alerter to use (slack, email, etc.)
        timeout: Timeout in seconds for approval response
        max_queries: Maximum number of queries to approve

    Returns:
        ApprovalDecision object with approval status and selected queries
    """
    start_time = time.time()

    # Limit queries to max_queries
    limited_queries = recommended_queries[:max_queries]

    # If approval not required, auto-approve all
    if not require_approval:
        logger.info(
            f"Auto-approving {len(limited_queries)} recommended queries (approval not required)"
        )

        # Log metadata for auto-approval
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "approval_decision": {
                    "execution_time_seconds": execution_time,
                    "approval_required": False,
                    "approval_method": "AUTO_APPROVED",
                    "num_queries_recommended": len(recommended_queries),
                    "num_queries_approved": len(limited_queries),
                    "max_queries_allowed": max_queries,
                    "approval_status": "approved",
                    "wait_time_seconds": 0,
                }
            }
        )

        return ApprovalDecision(
            approved=True,
            selected_queries=limited_queries,
            approval_method="AUTO_APPROVED",
            reviewer_notes="Approval not required by configuration",
        )

    # If no queries to approve, skip
    if not limited_queries:
        logger.info("No additional queries recommended")

        # Log metadata for no queries
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "approval_decision": {
                    "execution_time_seconds": execution_time,
                    "approval_required": require_approval,
                    "approval_method": "NO_QUERIES",
                    "num_queries_recommended": 0,
                    "num_queries_approved": 0,
                    "max_queries_allowed": max_queries,
                    "approval_status": "skipped",
                    "wait_time_seconds": 0,
                }
            }
        )

        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="NO_QUERIES",
            reviewer_notes="No additional queries recommended",
        )

    # Prepare approval request
    progress_summary = summarize_research_progress_from_artifacts(
        synthesis_data, analysis_data
    )

    # Extract critique points from analysis data
    critique_points = []
    if analysis_data.critique_summary:
        # Convert critique summary to list of dicts for compatibility
        for i, critique in enumerate(
            analysis_data.critique_summary.split("\n")
        ):
            if critique.strip():
                critique_points.append(
                    {
                        "issue": critique.strip(),
                        "importance": "high" if i < 3 else "medium",
                    }
                )

    message = format_approval_request(
        main_query=query_context.main_query,
        progress_summary=progress_summary,
        critique_points=critique_points,
        proposed_queries=limited_queries,
        timeout=timeout,
    )

    # Log the approval request for visibility
    logger.info("=" * 80)
    logger.info("APPROVAL REQUEST:")
    logger.info(message)
    logger.info("=" * 80)

    try:
        # Get the alerter from the active stack
        client = Client()
        alerter = client.active_stack.alerter

        if not alerter:
            logger.warning("No alerter configured in stack, auto-approving")

            # Log metadata for no alerter scenario
            execution_time = time.time() - start_time
            log_metadata(
                metadata={
                    "approval_decision": {
                        "execution_time_seconds": execution_time,
                        "approval_required": require_approval,
                        "approval_method": "NO_ALERTER_AUTO_APPROVED",
                        "alerter_type": "none",
                        "num_queries_recommended": len(recommended_queries),
                        "num_queries_approved": len(limited_queries),
                        "max_queries_allowed": max_queries,
                        "approval_status": "auto_approved",
                        "wait_time_seconds": 0,
                    }
                }
            )

            return ApprovalDecision(
                approved=True,
                selected_queries=limited_queries,
                approval_method="NO_ALERTER_AUTO_APPROVED",
                reviewer_notes="No alerter configured - auto-approved",
            )

        # Use the alerter's ask method for interactive approval
        try:
            # Send the message to Discord and wait for response
            logger.info(
                f"Sending approval request to {alerter.flavor} alerter"
            )

            # Format message for Discord (Discord has message length limits)
            discord_message = (
                f"**Research Approval Request**\n\n{message[:1900]}"
            )
            if len(message) > 1900:
                discord_message += (
                    "\n\n*(Message truncated due to Discord limits)*"
                )

            # Add instructions for Discord responses
            discord_message += "\n\n**How to respond:**\n"
            discord_message += "✅ Type `yes`, `approve`, `ok`, or `LGTM` to approve ALL queries\n"
            discord_message += "❌ Type `no`, `skip`, `reject`, or `decline` to skip additional research\n"
            discord_message += f"⏱️ Response timeout: {timeout} seconds"

            # Use the ask method to get user response
            logger.info("Waiting for approval response from Discord...")
            wait_start_time = time.time()
            approved = alerter.ask(discord_message)
            wait_end_time = time.time()
            wait_time = wait_end_time - wait_start_time

            logger.info(
                f"Received Discord response: {'approved' if approved else 'rejected'}"
            )

            if approved:
                # Log metadata for approved decision
                execution_time = time.time() - start_time
                log_metadata(
                    metadata={
                        "approval_decision": {
                            "execution_time_seconds": execution_time,
                            "approval_required": require_approval,
                            "approval_method": "DISCORD_APPROVED",
                            "alerter_type": alerter_type,
                            "num_queries_recommended": len(
                                recommended_queries
                            ),
                            "num_queries_approved": len(limited_queries),
                            "max_queries_allowed": max_queries,
                            "approval_status": "approved",
                            "wait_time_seconds": wait_time,
                            "timeout_configured": timeout,
                        }
                    }
                )

                return ApprovalDecision(
                    approved=True,
                    selected_queries=limited_queries,
                    approval_method="DISCORD_APPROVED",
                    reviewer_notes="Approved via Discord",
                )
            else:
                # Log metadata for rejected decision
                execution_time = time.time() - start_time
                log_metadata(
                    metadata={
                        "approval_decision": {
                            "execution_time_seconds": execution_time,
                            "approval_required": require_approval,
                            "approval_method": "DISCORD_REJECTED",
                            "alerter_type": alerter_type,
                            "num_queries_recommended": len(
                                recommended_queries
                            ),
                            "num_queries_approved": 0,
                            "max_queries_allowed": max_queries,
                            "approval_status": "rejected",
                            "wait_time_seconds": wait_time,
                            "timeout_configured": timeout,
                        }
                    }
                )

                return ApprovalDecision(
                    approved=False,
                    selected_queries=[],
                    approval_method="DISCORD_REJECTED",
                    reviewer_notes="Rejected via Discord",
                )

        except Exception as e:
            logger.error(f"Failed to get approval from alerter: {e}")

            # Log metadata for alerter error
            execution_time = time.time() - start_time
            log_metadata(
                metadata={
                    "approval_decision": {
                        "execution_time_seconds": execution_time,
                        "approval_required": require_approval,
                        "approval_method": "ALERTER_ERROR",
                        "alerter_type": alerter_type,
                        "num_queries_recommended": len(recommended_queries),
                        "num_queries_approved": 0,
                        "max_queries_allowed": max_queries,
                        "approval_status": "error",
                        "error_message": str(e),
                    }
                }
            )

            return ApprovalDecision(
                approved=False,
                selected_queries=[],
                approval_method="ALERTER_ERROR",
                reviewer_notes=f"Failed to get approval: {str(e)}",
            )

    except Exception as e:
        logger.error(f"Approval step failed: {e}")

        # Log metadata for general error
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "approval_decision": {
                    "execution_time_seconds": execution_time,
                    "approval_required": require_approval,
                    "approval_method": "ERROR",
                    "num_queries_recommended": len(recommended_queries),
                    "num_queries_approved": 0,
                    "max_queries_allowed": max_queries,
                    "approval_status": "error",
                    "error_message": str(e),
                }
            }
        )

        # Add tag to the approval decision artifact
        # add_tags(tags=["hitl"], artifact_name="approval_decision", infer_artifact=True)

        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="ERROR",
            reviewer_notes=f"Approval failed: {str(e)}",
        )
