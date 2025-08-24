"""
Generate comprehensive pipeline report.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, Optional

from zenml import step
from zenml.logger import get_logger
from zenml.types import MarkdownString

logger = get_logger(__name__)


@step
def report(
    workspace_dir: Path,
    commit_sha: str,
    prompt_used: str,
    agent_results: Dict,
    baseline_results: Optional[Dict],
) -> Annotated[MarkdownString, "final_report"]:
    """
    Generate comprehensive markdown report for pipeline execution.

    Args:
        workspace_dir: Workspace directory path
        commit_sha: Git commit SHA
        prompt_used: Prompt template used
        agent_results: Agent test results
        baseline_results: Baseline test results (optional)

    Returns:
        Path to generated markdown report
    """
    logger.info("Generating pipeline execution report")

    # Create report file
    report_file = (
        Path(tempfile.mkdtemp(prefix="qualityflow_report_")) / "report.md"
    )

    # Evaluate coverage metrics first
    evaluation_metrics = _evaluate_coverage_metrics(
        agent_results, baseline_results, commit_sha
    )

    # Generate report content
    report_content = _generate_report_content(
        workspace_dir,
        commit_sha,
        prompt_used,
        agent_results,
        baseline_results,
        evaluation_metrics,
    )

    # Write report file
    with open(report_file, "w") as f:
        f.write(report_content)

    logger.info(f"Report generated: {report_file}")

    # Return as MarkdownString for dashboard visualization
    return MarkdownString(report_content)


def _evaluate_coverage_metrics(
    agent_results: Dict,
    baseline_results: Optional[Dict],
    commit_sha: str,
) -> Dict:
    """Evaluate coverage metrics and compare agent vs baseline approaches."""

    # Extract agent metrics - use actual values from test results
    coverage_total_agent = agent_results.get("coverage_total", 0.0)
    tests_passed_agent = agent_results.get("tests_passed", 0)
    tests_failed_agent = agent_results.get("tests_failed", 0)

    total_tests_agent = tests_passed_agent + tests_failed_agent
    pass_rate_agent = (
        tests_passed_agent / total_tests_agent
        if total_tests_agent > 0
        else 0.0
    )

    # Extract baseline metrics
    coverage_total_baseline = 0.0
    if baseline_results and not baseline_results.get("skipped", False):
        coverage_total_baseline = baseline_results.get("coverage_total", 0.0)

    # Compare agent vs baseline coverage
    coverage_improvement = coverage_total_agent - coverage_total_baseline

    # Analyze coverage quality
    pass_rate_quality = (
        "excellent"
        if pass_rate_agent > 0.95
        else "good"
        if pass_rate_agent > 0.8
        else "needs_improvement"
    )
    coverage_quality = (
        "excellent"
        if coverage_total_agent > 80
        else "good"
        if coverage_total_agent > 50
        else "needs_improvement"
    )

    return {
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


def _generate_report_content(
    workspace_dir: Path,
    commit_sha: str,
    prompt_used: str,
    agent_results: Dict,
    baseline_results: Optional[Dict],
    evaluation_metrics: Dict,
) -> str:
    """Generate markdown report content."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Header
    report = f"""# QualityFlow Pipeline Report

Generated: {timestamp}
Commit: `{commit_sha}`
Workspace: `{workspace_dir}`

## Executive Summary

"""

    # Executive summary
    coverage_agent = evaluation_metrics.get("coverage_total_agent", 0.0)
    coverage_baseline = evaluation_metrics.get("coverage_total_baseline", 0.0)
    improvement = evaluation_metrics.get("coverage_improvement", 0.0)
    quality = evaluation_metrics.get("coverage_quality", "unknown")

    quality_emoji = (
        "🟢" if quality == "excellent" else "🟡" if quality == "good" else "🔴"
    )
    improvement_emoji = (
        "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
    )

    report += f"""{quality_emoji} **Coverage Quality**: {quality.upper()}
{improvement_emoji} **Agent vs Baseline**: {coverage_agent:.2f}% vs {coverage_baseline:.2f}% ({improvement:+.2f}%)  
🧪 **Tests**: {agent_results.get("tests_passed", 0)} passed, {agent_results.get("tests_failed", 0)} failed
📁 **Files**: {evaluation_metrics.get("files_analyzed", 0)} analyzed

"""

    # Agent results section
    report += """## Agent Test Results

"""

    if agent_results.get("skipped", False):
        report += "Agent tests were skipped.\n\n"
    else:
        report += f"""- **Tests Passed**: {agent_results.get("tests_passed", 0)}
- **Tests Failed**: {agent_results.get("tests_failed", 0)}
- **Pass Rate**: {evaluation_metrics.get("pass_rate_agent", 0.0):.1%}
- **Coverage**: {agent_results.get("coverage_total", 0.0):.2f}%
- **JUnit Report**: `{agent_results.get("junit_path", "N/A")}`
- **Coverage Report**: `{agent_results.get("coverage_path", "N/A")}`
- **Logs**: `{agent_results.get("logs_path", "N/A")}`

"""

    # Baseline results section (if available)
    if baseline_results and not baseline_results.get("skipped", False):
        report += """## Baseline Test Results

"""
        report += f"""- **Tests Passed**: {baseline_results.get("tests_passed", 0)}
- **Tests Failed**: {baseline_results.get("tests_failed", 0)}
- **Coverage**: {baseline_results.get("coverage_total", 0.0):.2f}%
- **JUnit Report**: `{baseline_results.get("junit_path", "N/A")}`
- **Coverage Report**: `{baseline_results.get("coverage_path", "N/A")}`

"""

    # Evaluation metrics section
    report += """## Coverage Analysis

"""

    pass_rate = evaluation_metrics.get("pass_rate_agent", 0.0)
    pass_quality = evaluation_metrics.get("pass_rate_quality", "unknown")

    report += f"""- **Agent Coverage**: {coverage_agent:.2f}% ({quality})
- **Baseline Coverage**: {coverage_baseline:.2f}%
- **Improvement**: {improvement:+.2f}%
- **Test Pass Rate**: {pass_rate:.1%} ({pass_quality})
- **Files Analyzed**: {evaluation_metrics.get("files_analyzed", 0)}

"""

    # Recommendations section
    report += """## Recommendations

"""
    if quality == "excellent":
        report += "🎉 **Excellent coverage!** Consider this approach for production use.\n"
    elif quality == "good":
        report += "👍 **Good coverage.** Consider tweaking prompts or selection strategy for improvement.\n"
    else:
        report += "⚠️ **Coverage needs improvement.** Try different prompts, models, or increase max_tests_per_file.\n"

    if improvement > 5:
        report += "📈 **Agent significantly outperforms baseline** - LLM approach is working well.\n"
    elif improvement > 0:
        report += "📊 **Agent slightly better than baseline** - room for optimization.\n"
    else:
        report += "📉 **Baseline performs as well or better** - review agent configuration.\n"

    # Configuration section
    report += """## Configuration

### Prompt Template
```
"""
    report += prompt_used[:500] + ("..." if len(prompt_used) > 500 else "")
    report += """
```

### File Coverage Details
"""

    coverage_by_file = agent_results.get("coverage_by_file", {})
    if coverage_by_file:
        report += "| File | Coverage |\n|------|----------|\n"
        for file_path, coverage_pct in sorted(coverage_by_file.items()):
            report += f"| `{file_path}` | {coverage_pct:.1f}% |\n"
    else:
        report += "No file-level coverage data available.\n"

    report += """

---
*Generated by QualityFlow - Production-ready test generation with ZenML*
"""

    return report
