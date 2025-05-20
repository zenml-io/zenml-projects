import json
import logging
from typing import Any, Dict

import openai
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)

logger = logging.getLogger(__name__)

# System prompt for information synthesis
SYNTHESIS_PROMPT = """
You are a Deep Research assistant. You will be given a sub-question and search results related to this question.
Your task is to synthesize the information from these sources to create a comprehensive, accurate, and balanced answer.

During synthesis, you should:
1. Validate the information for accuracy and reliability
2. Remove redundancies while preserving important details
3. Identify and note any contradictions or disagreements between sources
4. Organize the information in a coherent and logical structure
5. Cite specific sources when presenting key facts or claims

Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "object",
  "properties": {
    "synthesized_answer": {"type": "string"},
    "key_sources": {
      "type": "array",
      "items": {"type": "string"}
    },
    "confidence_level": {"type": "string", "enum": ["high", "medium", "low"]},
    "information_gaps": {"type": "string"}
  }
}
</OUTPUT JSON SCHEMA>

Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


def _synthesize_information(
    synthesis_input: Dict[str, Any],
    openai_client: openai.OpenAI,
    model: str,
    system_prompt: str,
) -> Dict[str, Any]:
    """Synthesize information from search results for a sub-question.

    Args:
        synthesis_input: Dictionary with sub-question, search results, and sources
        openai_client: OpenAI client
        model: Model to use
        system_prompt: System prompt for the LLM

    Returns:
        Dictionary with synthesized information
    """
    sub_question_for_log = synthesis_input.get(
        "sub_question", "unknown question"
    )
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(synthesis_input)},
            ],
        )

        # Defensive access to content
        llm_content_str = None
        if response and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice and choice.message:
                llm_content_str = choice.message.content

        if llm_content_str is None:
            logger.warning(
                f"LLM response content is missing or empty for '{sub_question_for_log}'."
            )
            return {
                "synthesized_answer": f"Synthesis failed due to missing LLM content for '{sub_question_for_log}'.",
                "key_sources": synthesis_input.get("sources", [])[:1],
                "confidence_level": "low",
                "information_gaps": "LLM did not provide content for synthesis.",
            }

        processed_content = remove_reasoning_from_output(llm_content_str)
        processed_content = clean_json_tags(processed_content)

        result = safe_json_loads(processed_content)

        if (
            not result
            or not isinstance(result, dict)
            or "synthesized_answer" not in result
        ):
            logger.warning(
                f"Failed to parse LLM response or 'synthesized_answer' missing for '{sub_question_for_log}'. "
                f"Content after cleaning (first 200 chars): '{processed_content[:200]}...'"
            )
            return {
                "synthesized_answer": f"Synthesis for '{sub_question_for_log}' failed due to parsing error or missing field.",
                "key_sources": synthesis_input.get("sources", [])[:1],
                "confidence_level": "low",
                "information_gaps": "LLM response parsing failed or critical fields were missing.",
            }

        return result

    except Exception as e:
        logger.error(
            f"Error synthesizing information for '{sub_question_for_log}': {e}",
            exc_info=True,
        )
        return {
            "synthesized_answer": f"Synthesis failed due to an unexpected error for '{sub_question_for_log}': {str(e)}",
            "key_sources": synthesis_input.get("sources", [])[:1],
            "confidence_level": "low",
            "information_gaps": "An unexpected technical error occurred during synthesis.",
        }
