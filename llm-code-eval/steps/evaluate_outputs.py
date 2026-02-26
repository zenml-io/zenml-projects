"""Step: Evaluate inference results using LLM-as-judge.

Sends each (prompt, canonical_solution, generated_code) triple to a
judge model that scores correctness, style, and completeness on 1-5.
"""

from __future__ import annotations

import logging
from typing import Annotated

from zenml import log_metadata, step

from utils.litellm_utils import (
    configure_litellm,
    litellm_completion_with_metrics,
    parse_json_safe,
)
from utils.scoring import (
    EvaluationResult,
    InferenceResult,
    JudgeScore,
    ModelAggregate,
)

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating Python function completions.

Score each completion on three dimensions (1-5 scale):

CORRECTNESS (1-5):
1 = Completely wrong, won't run or produces wrong output
2 = Major bugs, handles <50% of cases correctly
3 = Works for basic cases, fails edge cases
4 = Mostly correct, minor issues
5 = Fully correct, handles all cases

STYLE (1-5):
1 = Unreadable, non-Pythonic
2 = Poor style, hard to follow
3 = Acceptable but not clean
4 = Clean, follows conventions
5 = Exemplary, idiomatic Python

COMPLETENESS (1-5):
1 = Incomplete, missing core logic
2 = Partial implementation
3 = Core logic present, missing edge cases
4 = Handles most cases
5 = Comprehensive, production-ready

Return ONLY valid JSON:
{"correctness": X, "style": X, "completeness": X, "reasoning": "Brief explanation"}"""

JUDGE_USER_TEMPLATE = """## Function Specification
{prompt}

## Reference Solution
```python
{canonical_solution}
```

## Generated Solution
```python
{generated_code}
```

Evaluate the generated solution against the specification and reference."""


def _flatten_results(
    results: list[InferenceResult] | list[list[InferenceResult]],
) -> list[InferenceResult]:
    """Flatten potentially nested results from .product() fan-out."""
    flat: list[InferenceResult] = []
    for item in results:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _judge_single(
    result: InferenceResult,
    judge_model: str,
    temperature: float,
    max_tokens: int,
) -> EvaluationResult:
    """Score a single inference result with the judge model."""
    user_msg = JUDGE_USER_TEMPLATE.format(
        prompt=result.prompt,
        canonical_solution=result.canonical_solution,
        generated_code=result.generated_code,
    )

    try:
        raw, _ = litellm_completion_with_metrics(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            metadata={
                "project": "llm-code-eval",
                "task_id": result.task_id,
                "judge": True,
            },
        )

        parsed = parse_json_safe(raw)
        if parsed:
            score = JudgeScore(
                correctness=max(1, min(5, int(parsed.get("correctness", 1)))),
                style=max(1, min(5, int(parsed.get("style", 1)))),
                completeness=max(
                    1, min(5, int(parsed.get("completeness", 1)))
                ),
                reasoning=str(parsed.get("reasoning", ""))[:500],
            )
        else:
            score = JudgeScore(
                correctness=1,
                style=1,
                completeness=1,
                reasoning=f"Failed to parse judge output: {raw[:200]}",
            )
    except Exception as e:
        logger.error(
            "Judge failed for %s / %s: %s",
            result.model,
            result.task_id,
            e,
        )
        score = JudgeScore(
            correctness=1,
            style=1,
            completeness=1,
            reasoning=f"Judge error: {e}",
        )

    return EvaluationResult(
        task_id=result.task_id,
        model=result.model,
        judge_model=judge_model,
        score=score,
    )


def _compute_aggregates(
    results: list[InferenceResult],
    evaluations: list[EvaluationResult],
    models: list[str],
) -> dict[str, ModelAggregate]:
    """Compute per-model aggregate metrics."""
    aggregates: dict[str, ModelAggregate] = {}

    for model in models:
        model_results = [r for r in results if r.model == model]
        model_evals = [e for e in evaluations if e.model == model]

        if not model_results:
            continue

        num = len(model_evals) or 1
        aggregates[model] = ModelAggregate(
            model=model,
            num_tasks=len(model_results),
            avg_correctness=sum(
                e.score.correctness for e in model_evals
            ) / num,
            avg_style=sum(e.score.style for e in model_evals) / num,
            avg_completeness=sum(
                e.score.completeness for e in model_evals
            ) / num,
            avg_latency_seconds=sum(
                r.metrics.latency_seconds for r in model_results
            ) / len(model_results),
            total_cost_usd=sum(
                r.metrics.cost_usd or 0.0 for r in model_results
            ),
            error_count=sum(
                1 for r in model_results if r.metrics.error
            ),
        )

    return aggregates


@step
def evaluate_outputs(
    inference_results: list[InferenceResult],
    models: list[str],
    judge_model: str = "anthropic/claude-sonnet-4-5-20250514",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[
    Annotated[list[EvaluationResult], "evaluation_results"],
    Annotated[dict[str, ModelAggregate], "model_aggregates"],
]:
    """Evaluate all inference results using LLM-as-judge.

    Accepts potentially nested results from .product() and flattens
    them before evaluation.
    """
    configure_litellm()

    flat_results = _flatten_results(inference_results)
    logger.info(
        "Evaluating %d inference results with judge=%s",
        len(flat_results),
        judge_model,
    )

    evaluations: list[EvaluationResult] = []
    for i, result in enumerate(flat_results):
        logger.info(
            "Judging %d/%d: %s on %s",
            i + 1,
            len(flat_results),
            result.model,
            result.task_id,
        )
        ev = _judge_single(result, judge_model, temperature, max_tokens)
        evaluations.append(ev)

    aggregates = _compute_aggregates(flat_results, evaluations, models)

    # Log aggregates as metadata for observability
    for model, agg in aggregates.items():
        safe_key = model.replace("/", "_").replace(".", "_")
        log_metadata(
            metadata={
                f"{safe_key}/avg_correctness": agg.avg_correctness,
                f"{safe_key}/avg_style": agg.avg_style,
                f"{safe_key}/avg_completeness": agg.avg_completeness,
                f"{safe_key}/avg_latency": agg.avg_latency_seconds,
                f"{safe_key}/total_cost": agg.total_cost_usd,
            }
        )

    log_metadata(
        metadata={
            "total_evaluations": len(evaluations),
            "models_evaluated": list(aggregates.keys()),
        }
    )

    return evaluations, aggregates
