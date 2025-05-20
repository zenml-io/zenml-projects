import logging
from typing import Any, Dict

import openai
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)

logger = logging.getLogger(__name__)

# System prompt for generating search queries
SEARCH_QUERY_PROMPT = """
You are a Deep Research assistant. Your task is to create an effective web search query for the given research sub-question.

A good search query should:
1. Be concise and focused
2. Use specific keywords related to the sub-question
3. Be formulated to retrieve accurate and relevant information
4. Avoid ambiguous terms or overly broad language

Consider what would most effectively retrieve information from search engines to answer the specific sub-question.

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "search_query": {"type": "string"},
    "reasoning": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


def _generate_search_query(
    sub_question: str,
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Generate an optimized search query for a sub-question.

    Args:
        sub_question: The sub-question to generate a search query for
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM

    Returns:
        Dictionary with search query and reasoning
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sub_question},
            ],
        )

        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        result = safe_json_loads(content)

        if not result or "search_query" not in result:
            # Fallback if parsing fails
            return {"search_query": sub_question, "reasoning": ""}

        return result

    except Exception as e:
        logger.error(f"Error generating search query: {e}")
        return {"search_query": sub_question, "reasoning": ""}
