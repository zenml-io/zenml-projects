import json
import logging
import os
from typing import Any, Dict, List, Optional

import openai
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)

logger = logging.getLogger(__name__)


def get_sambanova_client(
    base_url: str = "https://api.sambanova.ai/v1",
) -> openai.OpenAI:
    """Get an OpenAI client configured for SambaNova.

    Args:
        base_url: SambaNova API base URL

    Returns:
        Configured OpenAI client for SambaNova

    Raises:
        ValueError: If SAMBANOVA_API_KEY environment variable is not set
    """
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    return openai.OpenAI(api_key=sambanova_api_key, base_url=base_url)


def run_llm_completion(
    prompt: str,
    system_prompt: str,
    client: openai.OpenAI,
    model: str,
    clean_output: bool = True,
) -> str:
    """Run an LLM completion with standard error handling and output cleaning.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        client: OpenAI client instance
        model: Model to use for completion
        clean_output: Whether to clean reasoning and JSON tags from output

    Returns:
        Processed LLM output
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        # Defensive access to content
        content = None
        if response and response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice and choice.message:
                content = choice.message.content

        if content is None:
            logger.warning("LLM response content is missing or empty.")
            return ""

        if clean_output:
            content = remove_reasoning_from_output(content)
            content = clean_json_tags(content)

        return content
    except Exception as e:
        logger.error(f"Error in LLM completion: {e}")
        return ""


def get_structured_llm_output(
    prompt: str,
    system_prompt: str,
    client: openai.OpenAI,
    model: str,
    fallback_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get structured JSON output from an LLM with error handling.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        client: OpenAI client instance
        model: Model to use for completion
        fallback_response: Fallback response if parsing fails

    Returns:
        Parsed JSON response or fallback
    """
    try:
        content = run_llm_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            client=client,
            model=model,
            clean_output=True,
        )

        if not content:
            logger.warning("Empty content returned from LLM")
            return fallback_response if fallback_response is not None else {}

        result = safe_json_loads(content)

        if not result and fallback_response is not None:
            return fallback_response

        return result
    except Exception as e:
        logger.error(f"Error processing structured LLM output: {e}")
        return fallback_response if fallback_response is not None else {}


def is_text_relevant(text1: str, text2: str, min_word_length: int = 4) -> bool:
    """Determine if two pieces of text are relevant to each other.

    Args:
        text1: First text
        text2: Second text
        min_word_length: Minimum length of words to check for overlap

    Returns:
        True if the texts are relevant to each other
    """
    if not text1 or not text2:
        return False

    return (
        text1.lower() in text2.lower()
        or text2.lower() in text1.lower()
        or any(
            word
            for word in text1.lower().split()
            if len(word) > min_word_length and word in text2.lower()
        )
    )


def find_most_relevant_string(
    target: str,
    options: List[str],
    client: Optional[openai.OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Find the most relevant string from a list of options using simple text matching.

    If client and model are provided, uses LLM to determine relevance.

    Args:
        target: The target string to find relevance for
        options: List of string options to check against
        client: Optional OpenAI client for more accurate relevance
        model: Model to use if client is provided

    Returns:
        The most relevant string, or None if no relevant options
    """
    if not options:
        return None

    if len(options) == 1:
        return options[0]

    # If LLM client is provided, use it for more accurate matching
    if client and model:
        try:
            system_prompt = "You are a research assistant."
            prompt = f"""Given the text: "{target}"
Which of the following options is most relevant to this text?
{options}

Respond with only the exact text of the most relevant option."""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            answer = response.choices[0].message.content.strip()

            # Check if the answer is one of the options
            if answer in options:
                return answer

            # If not an exact match, find the closest one
            for option in options:
                if option in answer or answer in option:
                    return option

        except Exception as e:
            logger.error(f"Error finding relevant string with LLM: {e}")

    # Simple relevance check - find exact matches first
    for option in options:
        if target.lower() == option.lower():
            return option

    # Then check partial matches
    for option in options:
        if is_text_relevant(target, option):
            return option

    # Return the first option as a fallback
    return options[0]


def synthesize_information(
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
