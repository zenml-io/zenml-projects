import logging
import os
import openai
import json
from typing import Annotated, Dict, Any
from zenml import step

from materializers.research_state_materializer import ResearchStateMaterializer
from utils.data_models import ResearchState, SynthesizedInfo
from utils.helper_functions import (
    remove_reasoning_from_output,
    clean_json_tags,
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


@step(output_materializers=ResearchStateMaterializer)
def information_validation_synthesis_step(
    state: ResearchState,
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = SYNTHESIS_PROMPT,
) -> Annotated[ResearchState, "updated_state"]:
    """Validate and synthesize information from search results for each sub-question.

    Args:
        state: The current research state
        sambanova_base_url: SambaNova API base URL
        llm_model: The model to use for synthesis
        system_prompt: System prompt for the LLM

    Returns:
        Updated research state with synthesized information
    """
    logger.info(
        f"Synthesizing information for {len(state.sub_questions)} sub-questions"
    )

    # Get API key from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    # Dictionary to store synthesized results
    synthesized_info = {}

    # Process each sub-question
    for i, sub_question in enumerate(state.sub_questions):
        logger.info(
            f"Synthesizing information for sub-question {i + 1}/{len(state.sub_questions)}: {sub_question}"
        )

        # Get search results for this sub-question
        question_search_results = state.search_results.get(sub_question, [])

        if not question_search_results:
            logger.warning(
                f"No search results found for sub-question: {sub_question}"
            )
            synthesized_info[sub_question] = SynthesizedInfo(
                synthesized_answer=f"No information found for: {sub_question}",
                key_sources=[],
                confidence_level="low",
                information_gaps="No search results were available for synthesis.",
            )
            continue

        # Extract raw contents and URLs
        raw_contents = []
        sources = []
        for result in question_search_results:
            raw_contents.append(result.content)
            sources.append(result.url)

        # Prepare input for synthesis
        synthesis_input = {
            "sub_question": sub_question,
            "search_results": raw_contents,
            "sources": sources,
        }

        # Synthesize information
        synthesis_result = _synthesize_information(
            synthesis_input=synthesis_input,
            openai_client=openai_client,
            model=llm_model,
            system_prompt=system_prompt,
        )

        # Create SynthesizedInfo object
        synthesized_info[sub_question] = SynthesizedInfo(
            synthesized_answer=synthesis_result.get(
                "synthesized_answer", f"Synthesis for '{sub_question}' failed."
            ),
            key_sources=synthesis_result.get("key_sources", sources[:1]),
            confidence_level=synthesis_result.get("confidence_level", "low"),
            information_gaps=synthesis_result.get(
                "information_gaps",
                "Synthesis process encountered technical difficulties.",
            ),
        )

    logger.info(f"Completed information synthesis for all sub-questions")

    # Update the state with synthesized information
    state.update_synthesized_info(synthesized_info)

    return state


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
