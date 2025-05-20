import json
import logging
from typing import Any, Dict, Optional

import openai
from utils.llm_utils import get_structured_llm_output

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
    system_prompt: Optional[str] = None,
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
    if system_prompt is None:
        system_prompt = SYNTHESIS_PROMPT

    sub_question_for_log = synthesis_input.get(
        "sub_question", "unknown question"
    )

    # Define the fallback response
    fallback_response = {
        "synthesized_answer": f"Synthesis failed for '{sub_question_for_log}'.",
        "key_sources": synthesis_input.get("sources", [])[:1],
        "confidence_level": "low",
        "information_gaps": "An error occurred during the synthesis process.",
    }

    # Use the utility function to get structured output
    result = get_structured_llm_output(
        prompt=json.dumps(synthesis_input),
        system_prompt=system_prompt,
        client=openai_client,
        model=model,
        fallback_response=fallback_response,
    )

    return result
