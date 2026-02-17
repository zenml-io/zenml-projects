"""Pipeline: LLM Code Evaluation.

Evaluates multiple LLM models on HumanEval coding problems using
a fan-out/fan-in pattern with ZenML dynamic pipelines.

DAG:
  load_test_cases -> run_inference.product(test_cases x models) ->
  evaluate_outputs -> generate_report
"""

from __future__ import annotations

from zenml import pipeline

from steps.evaluate_outputs import evaluate_outputs
from steps.generate_report import generate_report
from steps.load_test_cases import load_test_cases
from steps.run_inference import run_inference


@pipeline(name="code_eval_pipeline", dynamic=True, enable_cache=True)
def code_eval_pipeline(
    models: list[str],
    judge_model: str = "anthropic/claude-sonnet-4-5-20250514",
    report_title: str = "LLM Code Evaluation Results",
) -> None:
    """Evaluate coding LLMs using HumanEval problems.

    Uses .product() for cartesian product fan-out: every test case
    is evaluated by every model in parallel.
    """
    # Step 1: Load curated test cases
    test_cases = load_test_cases()

    # Step 2: Fan-out — run inference for every (test_case, model) pair
    # .product() creates N_tests x N_models parallel step invocations
    all_results = run_inference.product(
        test_case=test_cases,
        model_name=models,
    )

    # Step 3: Evaluate all results with LLM-as-judge
    evaluation_results, model_aggregates = evaluate_outputs(
        inference_results=all_results,
        models=models,
        judge_model=judge_model,
    )

    # Step 4: Generate HTML comparison report
    generate_report(
        test_cases=test_cases,
        inference_results=all_results,
        evaluation_results=evaluation_results,
        model_aggregates=model_aggregates,
        models=models,
        report_title=report_title,
    )
