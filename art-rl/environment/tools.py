"""LangGraph tools for the email search agent."""

from dataclasses import asdict
from typing import Callable, List, Optional

from langchain_core.tools import tool

from environment.email_db import read_email, search_emails
from environment.models import FinalAnswer, Scenario


def create_email_tools(
    scenario: Scenario,
    db_path: str,
    on_final_answer: Optional[Callable[[FinalAnswer], None]] = None,
) -> List:
    """Create LangGraph tools for a specific scenario.

    The tools are scenario-specific because they need access to the
    inbox address and query date for filtering.

    Args:
        scenario: The current scenario being processed.
        db_path: Path to the email database.
        on_final_answer: Callback invoked when the agent provides a final answer.

    Returns:
        List of LangChain tools for the LangGraph agent.
    """

    @tool
    def search_inbox_tool(keywords: List[str]) -> List[dict]:
        """Search the inbox for emails matching the given keywords.

        Args:
            keywords: List of keywords to search for (uses AND logic).

        Returns:
            List of search results with message_id and snippet.
        """
        results = search_emails(
            inbox=scenario.inbox_address,
            keywords=keywords,
            db_path=db_path,
            sent_before=scenario.query_date,
        )
        return [asdict(result) for result in results]

    @tool
    def read_email_tool(message_id: str) -> Optional[dict]:
        """Read a specific email by message ID.

        Args:
            message_id: The unique identifier of the email to read.

        Returns:
            Email content as a dictionary, or None if not found.
        """
        email = read_email(message_id, db_path=db_path)
        if email:
            return email.model_dump()
        return None

    @tool
    def return_final_answer_tool(
        answer: str,
        reference_message_ids: List[str],
    ) -> dict:
        """Return the final answer with source references.

        Use this tool when you have found the answer to the user's question.

        Args:
            answer: The answer to the user's question.
            reference_message_ids: List of message IDs that support the answer.

        Returns:
            The final answer as a dictionary.
        """
        final_answer = FinalAnswer(
            answer=answer,
            source_ids=reference_message_ids,
        )
        if on_final_answer:
            on_final_answer(final_answer)
        return final_answer.model_dump()

    return [search_inbox_tool, read_email_tool, return_final_answer_tool]
