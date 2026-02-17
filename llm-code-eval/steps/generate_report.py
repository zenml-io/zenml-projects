"""Step: Generate HTML comparison report.

Assembles all results into a CodeEvalReport with embedded HTML
that the custom materializer renders in the ZenML dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated

from zenml import log_metadata, step

from utils.html_templates import build_full_report
from utils.scoring import (
    CodeEvalReport,
    EvaluationResult,
    InferenceResult,
    ModelAggregate,
    TestCase,
)

logger = logging.getLogger(__name__)


@step
def generate_report(
    test_cases: list[TestCase],
    inference_results: list[InferenceResult],
    evaluation_results: list[EvaluationResult],
    model_aggregates: dict[str, ModelAggregate],
    models: list[str],
    report_title: str = "LLM Code Evaluation Results",
    include_per_problem_breakdown: bool = True,
) -> Annotated[CodeEvalReport, "code_eval_report"]:
    """Build the final CodeEvalReport with embedded HTML visualization."""
    # Flatten if needed (in case of nested .product() outputs)
    flat_results: list[InferenceResult] = []
    for item in inference_results:
        if isinstance(item, list):
            flat_results.extend(item)
        else:
            flat_results.append(item)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    task_ids = sorted({tc.task_id for tc in test_cases})

    report = CodeEvalReport(
        report_title=report_title,
        generated_at=now,
        models=models,
        test_case_ids=task_ids,
        aggregates=model_aggregates,
        results=flat_results,
        evaluations=evaluation_results,
        report_html="",  # populated below
    )

    report.report_html = build_full_report(report)

    logger.info(
        "Generated report: %d models, %d test cases, HTML size=%d bytes",
        len(models),
        len(task_ids),
        len(report.report_html),
    )

    # Log key findings as metadata
    if model_aggregates:
        best_model = max(
            model_aggregates.values(),
            key=lambda a: a.avg_correctness,
        )
        log_metadata(
            metadata={
                "best_correctness_model": best_model.model,
                "best_correctness_score": best_model.avg_correctness,
                "total_models": len(models),
                "total_test_cases": len(task_ids),
            }
        )

    return report
