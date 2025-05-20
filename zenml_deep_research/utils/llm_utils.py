import json
import logging
import os
from typing import Any, Dict, List, Optional

import openai
from litellm import completion
from utils.helper_functions import (
    clean_json_tags,
    remove_reasoning_from_output,
    safe_json_loads,
)
from utils.prompts import SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


def get_sambanova_client(
    base_url: str = "https://api.sambanova.ai/v1",
) -> openai.OpenAI:
    """Get an OpenAI client configured for SambaNova.

    This function still returns an OpenAI client for backward compatibility,
    but new code should use the litellm functions directly.

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
    client: Optional[
        openai.OpenAI
    ] = None,  # Made optional for backward compatibility
    model: str = "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    clean_output: bool = True,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Run an LLM completion with standard error handling and output cleaning.

    Uses litellm for model inference.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        client: OpenAI client instance (optional, kept for backward compatibility)
        model: Model to use for completion (with provider prefix)
        clean_output: Whether to clean reasoning and JSON tags from output. When True, 
            this removes any reasoning sections marked with </think> tags and strips JSON 
            code block markers.
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling value

    Returns:
        str: Processed LLM output with optional cleaning applied
    """
    try:
        # Ensure model name has provider prefix
        if not any(
            model.startswith(prefix + "/")
            for prefix in [
                "sambanova",
                "openai",
                "anthropic",
                "meta",
                "google",
                "aws",
            ]
        ):
            # Add sambanova prefix if not specified (assuming default provider)
            model = f"sambanova/{model}"

        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
    client: Optional[
        openai.OpenAI
    ] = None,  # Made optional for backward compatibility
    model: str = "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    fallback_response: Optional[Dict[str, Any]] = None,
    max_tokens: int = 1000,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    """Get structured JSON output from an LLM with error handling.

    Uses litellm for model inference.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        client: OpenAI client instance (optional, kept for backward compatibility)
        model: Model to use for completion (with provider prefix)
        fallback_response: Fallback response if parsing fails
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling value

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
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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

    Relevance is determined by checking if one text is contained within the other,
    or if they share significant words (words longer than min_word_length).
    This is a simple heuristic approach that checks for:
    1. Complete containment (one text string inside the other)
    2. Shared significant words (words longer than min_word_length)

    Args:
        text1: First text to compare
        text2: Second text to compare
        min_word_length: Minimum length of words to check for shared content

    Returns:
        bool: True if the texts are deemed relevant to each other based on the criteria
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
    client: Optional[openai.OpenAI] = None,  # Kept for backward compatibility
    model: Optional[str] = "sambanova/Llama-4-Maverick-17B-128E-Instruct",
) -> Optional[str]:
    """Find the most relevant string from a list of options using simple text matching.

    If model is provided, uses litellm to determine relevance.

    Args:
        target: The target string to find relevance for
        options: List of string options to check against
        client: Optional OpenAI client (kept for backward compatibility)
        model: Model to use for matching (with provider prefix)

    Returns:
        The most relevant string, or None if no relevant options
    """
    if not options:
        return None

    if len(options) == 1:
        return options[0]

    # If model is provided, use litellm for more accurate matching
    if model:
        try:
            # Ensure model name has provider prefix
            if not any(
                model.startswith(prefix + "/")
                for prefix in [
                    "sambanova",
                    "openai",
                    "anthropic",
                    "meta",
                    "google",
                    "aws",
                ]
            ):
                # Add sambanova prefix if not specified (assuming default provider)
                model = f"sambanova/{model}"

            system_prompt = "You are a research assistant."
            prompt = f"""Given the text: "{target}"
Which of the following options is most relevant to this text?
{options}

Respond with only the exact text of the most relevant option."""

            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.2,
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
    openai_client: Optional[
        openai.OpenAI
    ] = None,  # Made optional for backward compatibility
    model: str = "sambanova/Llama-4-Maverick-17B-128E-Instruct",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Synthesize information from search results for a sub-question.

    Uses litellm for model inference.

    Args:
        synthesis_input: Dictionary with sub-question, search results, and sources
        openai_client: OpenAI client (optional, kept for backward compatibility)
        model: Model to use (with provider prefix)
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
        max_tokens=2000,  # Increased for synthesis
    )

    return result
