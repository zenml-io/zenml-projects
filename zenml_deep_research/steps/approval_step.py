import logging
from typing import Annotated

from utils.approval_utils import (
    format_approval_request,
    summarize_research_progress,
)
from utils.pydantic_models import ApprovalDecision, ReflectionOutput
from zenml import step
from zenml.client import Client

logger = logging.getLogger(__name__)


@step(enable_cache=False)  # Never cache approval decisions
def get_research_approval_step(
    reflection_output: ReflectionOutput,
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
        reflection_output: Output from the reflection generation step
        require_approval: Whether to require human approval
        alerter_type: Type of alerter to use (slack, email, etc.)
        timeout: Timeout in seconds for approval response
        max_queries: Maximum number of queries to approve

    Returns:
        ApprovalDecision object with approval status and selected queries
    """

    # Limit queries to max_queries
    limited_queries = reflection_output.recommended_queries[:max_queries]

    # If approval not required, auto-approve all
    if not require_approval:
        logger.info(
            f"Auto-approving {len(limited_queries)} recommended queries (approval not required)"
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
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="NO_QUERIES",
            reviewer_notes="No additional queries recommended",
        )

    # Prepare approval request
    progress_summary = summarize_research_progress(reflection_output.state)
    message = format_approval_request(
        main_query=reflection_output.state.main_query,
        progress_summary=progress_summary,
        critique_points=reflection_output.critique_summary,
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
            approved = alerter.ask(discord_message)

            logger.info(
                f"Received Discord response: {'approved' if approved else 'rejected'}"
            )

            if approved:
                return ApprovalDecision(
                    approved=True,
                    selected_queries=limited_queries,
                    approval_method="DISCORD_APPROVED",
                    reviewer_notes="Approved via Discord",
                )
            else:
                return ApprovalDecision(
                    approved=False,
                    selected_queries=[],
                    approval_method="DISCORD_REJECTED",
                    reviewer_notes="Rejected via Discord",
                )

        except Exception as e:
            logger.error(f"Failed to get approval from alerter: {e}")
            return ApprovalDecision(
                approved=False,
                selected_queries=[],
                approval_method="ALERTER_ERROR",
                reviewer_notes=f"Failed to get approval: {str(e)}",
            )

    except Exception as e:
        logger.error(f"Approval step failed: {e}")
        return ApprovalDecision(
            approved=False,
            selected_queries=[],
            approval_method="ERROR",
            reviewer_notes=f"Approval failed: {str(e)}",
        )
