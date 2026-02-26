"""Pydantic models for the code evaluation pipeline.

Defines the typed data structures that flow between pipeline steps:
TestCase -> InferenceResult -> EvaluationResult -> CodeEvalReport
"""

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single coding problem to evaluate."""

    task_id: str
    prompt: str
    canonical_solution: str
    entry_point: str
    test: str = ""
    difficulty: str = "medium"
    source: str = "humaneval"


class InferenceMetrics(BaseModel):
    """Performance metrics from a single LLM inference call."""

    latency_seconds: float
    tokens_input: int | None = None
    tokens_output: int | None = None
    tokens_total: int | None = None
    cost_usd: float | None = None
    error: str | None = None


class InferenceResult(BaseModel):
    """Output from running inference on a single (test_case, model) pair."""

    task_id: str
    model: str
    prompt: str
    generated_code: str
    canonical_solution: str
    metrics: InferenceMetrics


class JudgeScore(BaseModel):
    """LLM-as-judge scores for a single code completion."""

    correctness: int = Field(ge=1, le=5)
    style: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    reasoning: str = ""


class EvaluationResult(BaseModel):
    """Judge evaluation of a single inference result."""

    task_id: str
    model: str
    judge_model: str
    score: JudgeScore


class ModelAggregate(BaseModel):
    """Aggregated metrics across all test cases for a single model."""

    model: str
    num_tasks: int
    avg_correctness: float
    avg_style: float
    avg_completeness: float
    avg_latency_seconds: float
    total_cost_usd: float
    error_count: int = 0


class CodeEvalReport(BaseModel):
    """Complete evaluation report — the final pipeline artifact."""

    report_title: str
    generated_at: str
    models: list[str]
    test_case_ids: list[str]
    aggregates: dict[str, ModelAggregate]
    results: list[InferenceResult]
    evaluations: list[EvaluationResult]
    report_html: str
