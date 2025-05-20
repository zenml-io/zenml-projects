import logging
from typing import Any, Dict, Optional

import openai
from utils.search_utils import generate_search_query

logger = logging.getLogger(__name__)


# The function is kept for backward compatibility but delegates to the utility function
def _generate_search_query(
    sub_question: str,
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an optimized search query for a sub-question.

    This function is maintained for backward compatibility. It delegates to the
    utility function in search_utils.py.

    Args:
        sub_question: The sub-question to generate a search query for
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM

    Returns:
        Dictionary with search query and reasoning
    """
    return generate_search_query(
        sub_question=sub_question,
        openai_client=openai_client,
        model=model,
        system_prompt=system_prompt,
    )
