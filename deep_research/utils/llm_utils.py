import contextlib
import json
import logging
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional

import litellm
from utils.prompts import SYNTHESIS_PROMPT
from utils.tracking_config import configure_tracking_provider, get_tracking_metadata
from utils.weave_zenml_integration import log_weave_trace_to_zenml, add_weave_context_to_function
from zenml import get_step_context

# Import weave for optional decorators
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False

logger = logging.getLogger(__name__)

# This module uses litellm for all LLM interactions
# Models are specified with a provider prefix:
# - For Google Gemini via OpenRouter: "openrouter/google/gemini-2.0-flash-lite-001"
# - For direct Google Gemini API: "gemini/gemini-2.0-flash-lite-001"
# - For other providers: "sambanova/", "openai/", "anthropic/", "meta/", "aws/"
# ALL model names require a provider prefix

# Default tracking configuration - will be updated by configure_tracking_provider()
# Note: This will be overridden by configure_tracking_provider() in each step
litellm.callbacks = []


def remove_reasoning_from_output(output: str) -> str:
    """Remove the reasoning portion from LLM output.

    Args:
        output: Raw output from LLM that may contain reasoning

    Returns:
        Cleaned output without the reasoning section
    """
    if not output:
        return ""

    if "</think>" in output:
        return output.split("</think>")[-1].strip()
    return output.strip()


def clean_json_tags(text: str) -> str:
    """Clean JSON markdown tags from text.

    Args:
        text: Text with potential JSON markdown tags

    Returns:
        Cleaned text without JSON markdown tags
    """
    if not text:
        return ""

    cleaned = text.replace("```json\n", "").replace("\n```", "")
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return cleaned


def clean_markdown_tags(text: str) -> str:
    """Clean Markdown tags from text.

    Args:
        text: Text with potential markdown tags

    Returns:
        Cleaned text without markdown tags
    """
    if not text:
        return ""

    cleaned = text.replace("```markdown\n", "").replace("\n```", "")
    cleaned = cleaned.replace("```markdown", "").replace("```", "")
    return cleaned


def safe_json_loads(json_str: Optional[str]) -> Dict[str, Any]:
    """Safely parse JSON string.

    Args:
        json_str: JSON string to parse, can be None.

    Returns:
        Dict[str, Any]: Parsed JSON as dictionary or empty dict if parsing fails or input is None.
    """
    if json_str is None:
        # Optionally, log a warning here if None input is unexpected for certain call sites
        # logger.warning("safe_json_loads received None input.")
        return {}
    try:
        return json.loads(json_str)
    except (
        JSONDecodeError,
        TypeError,
    ):  # Catch TypeError if json_str is not a valid type for json.loads
        # Optionally, log the error and the problematic string (or its beginning)
        # logger.warning(f"Failed to decode JSON string: '{str(json_str)[:200]}...'", exc_info=True)
        return {}


# Conditional weave decorator
def _weave_op_if_available(func):
    """Conditionally apply weave.op decorator if weave is available."""
    if WEAVE_AVAILABLE:
        return weave.op()(func)
    return func

@_weave_op_if_available
def run_llm_completion(
    prompt: str,
    system_prompt: str,
    model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    clean_output: bool = True,
    max_tokens: int = 2000,  # Increased default token limit
    temperature: float = 0.2,
    top_p: float = 0.9,
    tracking_provider: str = "weave",
    project: str = "deep-research",
    tags: Optional[List[str]] = None,
) -> str:
    """Run an LLM completion with standard error handling and output cleaning.

    Uses litellm for model inference.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        model: Model to use for completion (with provider prefix)
        clean_output: Whether to clean reasoning and JSON tags from output. When True,
            this removes any reasoning sections marked with </think> tags and strips JSON
            code block markers.
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling value
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        project: Project name for LLM tracking
        tags: Optional list of tags for tracking. If provided, also converted to trace_metadata format.

    Returns:
        str: Processed LLM output with optional cleaning applied
    """
    try:
        # Ensure model name has provider prefix
        # Special handling for OpenRouter models which have a nested provider
        if model.startswith("openrouter/"):
            # OpenRouter models are valid (e.g., openrouter/google/gemini-2.0-flash-lite-001)
            pass
        elif not any(
            model.startswith(prefix + "/")
            for prefix in [
                "sambanova",
                "openai",
                "anthropic",
                "meta",
                "gemini",  # Direct Google Gemini API
                "aws",
            ]
        ):
            # Raise an error if no provider prefix is specified
            error_msg = f"Model '{model}' does not have a provider prefix. Please specify provider (e.g., 'gemini/{model}', 'openrouter/{model}')"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get pipeline run name and id for trace_name and trace_id if running in a step
        trace_name = None
        trace_id = None
        with contextlib.suppress(RuntimeError):
            context = get_step_context()
            trace_name = context.pipeline_run.name
            trace_id = str(context.pipeline_run.id)
        
        # Build metadata dict using the tracking configuration
        # For Weave, we don't pass metadata as it handles tracking automatically
        metadata = {}
        if tracking_provider.lower() != "weave":
            metadata = get_tracking_metadata(
                tracking_provider=tracking_provider,
                project_name=project,
                tags=tags,
                trace_name=trace_name,
                trace_id=trace_id,
            )
        
        # Add ZenML context for Weave tracking
        if tracking_provider.lower() == "weave" and WEAVE_AVAILABLE:
            weave_context = add_weave_context_to_function(
                "run_llm_completion",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tags=tags or []
            )

        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
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

        # Log to ZenML metadata for Weave tracking
        if tracking_provider.lower() == "weave" and WEAVE_AVAILABLE:
            log_weave_trace_to_zenml(
                operation_name="llm_completion",
                additional_metadata={
                    "model": model,
                    "prompt_length": len(prompt),
                    "response_length": len(content) if content else 0,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

        return content
    except Exception as e:
        logger.error(f"Error in LLM completion: {e}")
        return ""


@_weave_op_if_available
def get_structured_llm_output(
    prompt: str,
    system_prompt: str,
    model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    fallback_response: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2000,  # Increased default token limit for structured outputs
    temperature: float = 0.2,
    top_p: float = 0.9,
    tracking_provider: str = "weave",
    project: str = "deep-research",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Get structured JSON output from an LLM with error handling.

    Uses litellm for model inference.

    Args:
        prompt: User prompt for the LLM
        system_prompt: System prompt for the LLM
        model: Model to use for completion (with provider prefix)
        fallback_response: Fallback response if parsing fails
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling value
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        project: Project name for LLM tracking
        tags: Optional list of tags for tracking. Defaults to ["structured_llm_output"] if None.

    Returns:
        Parsed JSON response or fallback
    """
    try:
        # Use provided tags or default to ["structured_llm_output"]
        if tags is None:
            tags = ["structured_llm_output"]

        content = run_llm_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            clean_output=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tracking_provider=tracking_provider,
            project=project,
            tags=tags,
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
    model: Optional[str] = "openrouter/google/gemini-2.0-flash-lite-001",
    tracking_provider: str = "weave",
    project: str = "deep-research",
    tags: Optional[List[str]] = None,
) -> Optional[str]:
    """Find the most relevant string from a list of options using simple text matching.

    If model is provided, uses litellm to determine relevance.

    Args:
        target: The target string to find relevance for
        options: List of string options to check against
        model: Model to use for matching (with provider prefix)
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        project: Project name for LLM tracking
        tags: Optional list of tags for tracking. Defaults to ["find_most_relevant_string"] if None.

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
            # Special handling for OpenRouter models which have a nested provider
            if model.startswith("openrouter/"):
                # OpenRouter models are valid (e.g., openrouter/google/gemini-2.0-flash-lite-001)
                pass
            elif not any(
                model.startswith(prefix + "/")
                for prefix in [
                    "sambanova",
                    "openai",
                    "anthropic",
                    "meta",
                    "gemini",  # Direct Google Gemini API
                    "aws",
                ]
            ):
                # Raise an error if no provider prefix is specified
                error_msg = f"Model '{model}' does not have a provider prefix. Please specify provider (e.g., 'gemini/{model}', 'openrouter/{model}')"
                logger.error(error_msg)
                raise ValueError(error_msg)

            system_prompt = "You are a research assistant."
            prompt = f"""Given the text: "{target}"
Which of the following options is most relevant to this text?
{options}

Respond with only the exact text of the most relevant option."""

            # Get pipeline run name and id for trace_name and trace_id if running in a step
            trace_name = None
            trace_id = None
            try:
                context = get_step_context()
                trace_name = context.pipeline_run.name
                trace_id = str(context.pipeline_run.id)
            except RuntimeError:
                # Not running in a step context
                pass

            # Use provided tags or default to ["find_most_relevant_string"]
            if tags is None:
                tags = ["find_most_relevant_string"]

            # Build metadata dict using the tracking configuration
            # For Weave, we don't pass metadata as it handles tracking automatically
            metadata = {}
            if tracking_provider.lower() != "weave":
                metadata = get_tracking_metadata(
                    tracking_provider=tracking_provider,
                    project_name=project,
                    tags=tags,
                    trace_name=trace_name,
                    trace_id=trace_id,
                )

            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.2,
                metadata=metadata,
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


@_weave_op_if_available
def synthesize_information(
    synthesis_input: Dict[str, Any],
    model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    system_prompt: Optional[str] = None,
    tracking_provider: str = "weave",
    project: str = "deep-research",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Synthesize information from search results for a sub-question.

    Uses litellm for model inference.

    Args:
        synthesis_input: Dictionary with sub-question, search results, and sources
        model: Model to use (with provider prefix)
        system_prompt: System prompt for the LLM
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        project: Project name for LLM tracking
        tags: Optional list of tags for tracking. Defaults to ["information_synthesis"] if None.

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

    # Use provided tags or default to ["information_synthesis"]
    if tags is None:
        tags = ["information_synthesis"]

    # Use the utility function to get structured output
    result = get_structured_llm_output(
        prompt=json.dumps(synthesis_input),
        system_prompt=system_prompt,
        model=model,
        fallback_response=fallback_response,
        max_tokens=3000,  # Increased for more detailed synthesis
        tracking_provider=tracking_provider,
        project=project,
        tags=tags,
    )

    return result
