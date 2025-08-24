"""
Evaluate coverage metrics and compare against baselines.
"""

from typing import Annotated, Dict, Optional
from zenml import step, Model
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def evaluate_coverage(
    agent_results: Dict,
    baseline_results: Optional[Dict],
    commit_sha: str,
) -> Annotated[Dict, "evaluation_metrics"]:
    """
    Evaluate coverage metrics and compare agent vs baseline approaches.
    
    Args:
        agent_results: Test results from agent-generated tests
        baseline_results: Test results from baseline tests (optional)
        commit_sha: Current commit SHA
        
    Returns:
        Evaluation metrics dictionary with coverage comparison
    """
    logger.info("Evaluating coverage metrics and computing deltas")
    
    # Extract agent metrics
    coverage_total_agent = agent_results.get("coverage_total", 0.0)
    tests_passed_agent = agent_results.get("tests_passed", 0)
    tests_failed_agent = agent_results.get("tests_failed", 0)
    
    total_tests_agent = tests_passed_agent + tests_failed_agent
    pass_rate_agent = tests_passed_agent / total_tests_agent if total_tests_agent > 0 else 0.0
    
    # Extract baseline metrics
    coverage_total_baseline = None
    if baseline_results and not baseline_results.get("skipped", False):
        coverage_total_baseline = baseline_results.get("coverage_total", 0.0)
    
    # Compare agent vs baseline coverage
    coverage_improvement = 0.0
    if coverage_total_baseline is not None:
        coverage_improvement = coverage_total_agent - coverage_total_baseline
    
    # Analyze coverage quality
    pass_rate_quality = "excellent" if pass_rate_agent > 0.95 else "good" if pass_rate_agent > 0.8 else "needs_improvement"
    coverage_quality = "excellent" if coverage_total_agent > 80 else "good" if coverage_total_agent > 50 else "needs_improvement"
    
    evaluation_metrics = {
        "coverage_total_agent": coverage_total_agent,
        "coverage_total_baseline": coverage_total_baseline,
        "coverage_improvement": coverage_improvement,
        "tests_passed_agent": tests_passed_agent,
        "tests_failed_agent": tests_failed_agent,
        "pass_rate_agent": pass_rate_agent,
        "pass_rate_quality": pass_rate_quality,
        "coverage_quality": coverage_quality,
        "commit_sha": commit_sha,
        "files_analyzed": len(agent_results.get("coverage_by_file", {})),
    }
    
    logger.info(f"Evaluation complete: agent_coverage={coverage_total_agent:.2f}%, baseline_coverage={coverage_total_baseline or 0:.2f}%, improvement={coverage_improvement:+.2f}%")
    
    return evaluation_metrics