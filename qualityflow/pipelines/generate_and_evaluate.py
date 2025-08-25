"""
QualityFlow experiment pipeline for test generation and evaluation.
"""

from steps.analyze_code import analyze_code
from steps.fetch_source import fetch_source
from steps.gen_tests_agent import gen_tests_agent
from steps.gen_tests_baseline import gen_tests_baseline
from steps.report import report
from steps.run_tests import run_tests
from steps.select_input import select_input
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline(name="generate_and_evaluate")
def generate_and_evaluate() -> None:
    """QualityFlow pipeline for generating and evaluating tests.

    Simple, focused pipeline:
    1. Analyze code to find files needing tests
    2. Generate tests using LLM and baseline approaches
    3. Run tests and measure coverage
    4. Report results for comparison
    """
    # Step 1: Resolve source specification
    spec = select_input()

    # Step 2: Fetch and materialize workspace
    workspace_dir, commit_sha = fetch_source(spec)

    # Step 3: Analyze and select code files
    code_summary = analyze_code(workspace_dir, commit_sha, spec)

    # Step 4: Generate tests using LLM agent
    agent_tests_dir, test_summary = gen_tests_agent(
        workspace_dir, code_summary
    )

    # Step 5: Generate baseline tests (optional)
    baseline_tests_dir = gen_tests_baseline(workspace_dir, code_summary)

    # Step 6: Run agent tests
    agent_results = run_tests(workspace_dir, agent_tests_dir, label="agent")

    # Step 7: Run baseline tests (if available)
    baseline_results = run_tests(
        workspace_dir, baseline_tests_dir, label="baseline"
    )

    # Step 8: Generate comprehensive report (includes evaluation)
    report(
        workspace_dir,
        commit_sha,
        test_summary,
        agent_results,
        baseline_results,
    )
