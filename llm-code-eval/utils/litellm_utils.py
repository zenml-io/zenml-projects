"""LiteLLM utilities for inference and cost tracking.

Wraps litellm.completion with latency measurement, token extraction,
cost computation, and optional Langfuse callback integration.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import litellm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.scoring import InferenceMetrics

logger = logging.getLogger(__name__)


def configure_litellm(
    project: str = "llm-code-eval",
    tags: list[str] | None = None,
) -> None:
    """Set up LiteLLM callbacks (Langfuse) if env vars are present."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        litellm.callbacks = ["langfuse"]
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
        logger.info("Langfuse callbacks enabled for LiteLLM")
    else:
        logger.info(
            "Langfuse env vars not set — running without tracing"
        )

    litellm.drop_params = True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(
        (litellm.exceptions.RateLimitError, litellm.exceptions.Timeout)
    ),
    reraise=True,
)
def litellm_completion_with_metrics(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_s: float | None = 120,
    metadata: dict | None = None,
) -> tuple[str, InferenceMetrics]:
    """Call litellm.completion and return (content, metrics).

    Measures wall-clock latency, extracts token usage, and computes
    cost via litellm.completion_cost when available.
    """
    start = time.perf_counter()

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_s,
        metadata=metadata or {},
    )

    latency = time.perf_counter() - start

    # Extract token usage safely
    usage = getattr(response, "usage", None)
    tokens_input = getattr(usage, "prompt_tokens", None)
    tokens_output = getattr(usage, "completion_tokens", None)
    tokens_total = getattr(usage, "total_tokens", None)

    # Compute cost (returns None for unknown models)
    cost = None
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        logger.debug("Could not compute cost for model=%s", model)

    content = response.choices[0].message.content or ""

    metrics = InferenceMetrics(
        latency_seconds=round(latency, 3),
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_total,
        cost_usd=round(cost, 6) if cost is not None else None,
    )

    return content, metrics


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences and leading explanations from LLM output."""
    # Try to extract content from ```python ... ``` blocks
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    # If no fences, return as-is (stripped)
    return text.strip()


def parse_json_safe(text: str) -> dict | None:
    """Parse JSON from LLM output, handling code fences and minor issues."""
    # Strip code fences if present
    cleaned = strip_code_fences(text)
    # Also try stripping any leading/trailing non-JSON text
    # Find first { and last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM output: %s", text[:200])
        return None
