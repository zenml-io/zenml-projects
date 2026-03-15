"""Step: Run inference on a single (test_case, model) pair.

Uses LiteLLM for model abstraction — any model supported by LiteLLM
can be evaluated by changing the config YAML.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import log_metadata, step

from utils.litellm_utils import (
    configure_litellm,
    litellm_completion_with_metrics,
    strip_code_fences,
)
from utils.scoring import InferenceMetrics, InferenceResult, TestCase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Complete the following Python function. "
    "Return only the function body code, no explanation or markdown."
)


@step
def run_inference(
    test_case: TestCase,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_s: float = 120,
) -> Annotated[InferenceResult, "inference_result"]:
    """Generate a code completion for one test case with one model.

    This step is designed to be used with .product() for fan-out
    across test_cases x models.
    """
    configure_litellm()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": test_case.prompt},
    ]

    try:
        raw_output, metrics = litellm_completion_with_metrics(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            metadata={
                "project": "llm-code-eval",
                "task_id": test_case.task_id,
            },
        )
        generated_code = strip_code_fences(raw_output)
    except Exception as e:
        logger.error(
            "Inference failed for %s on %s: %s",
            model_name,
            test_case.task_id,
            e,
        )
        generated_code = f"# ERROR: {e}"
        metrics = InferenceMetrics(
            latency_seconds=0.0,
            error=str(e),
        )

    log_metadata(
        metadata={
            "model": model_name,
            "task_id": test_case.task_id,
            "latency_seconds": metrics.latency_seconds,
            "tokens_total": metrics.tokens_total,
            "cost_usd": metrics.cost_usd,
        }
    )

    return InferenceResult(
        task_id=test_case.task_id,
        model=model_name,
        prompt=test_case.prompt,
        generated_code=generated_code,
        canonical_solution=test_case.canonical_solution,
        metrics=metrics,
    )
