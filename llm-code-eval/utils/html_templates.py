"""HTML template generation for the code evaluation report.

Keeps HTML generation separate from ZenML materializer logic so
templates can evolve independently of artifact persistence.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

from utils.scoring import (
    CodeEvalReport,
    EvaluationResult,
    InferenceResult,
    ModelAggregate,
)

REPORT_CSS = """
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           margin: 0; padding: 24px; background: #fafafa; color: #1a1a1a; }
    h1 { margin-bottom: 4px; }
    .subtitle { color: #666; margin-bottom: 24px; }
    .eval-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    .eval-table th, .eval-table td { border: 1px solid #ddd; padding: 10px 14px; text-align: center; }
    .eval-table th { background-color: #7c3aed; color: white; font-weight: 600; }
    .eval-table tr:nth-child(even) { background-color: #f8f7ff; }
    .eval-table tr:hover { background-color: #ede9fe; }
    .winner { background-color: #d1fae5 !important; font-weight: bold; }
    .metric-header { font-size: 0.78em; color: #ccc; display: block; }
    .model-name { text-align: left; font-family: monospace; font-size: 0.9em; }
    .section { margin: 32px 0; }
    .problem-card { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
                    padding: 16px; margin: 12px 0; }
    .problem-card h3 { margin-top: 0; color: #7c3aed; }
    .code-block { background: #1e1e2e; color: #cdd6f4; padding: 12px; border-radius: 6px;
                  overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.85em;
                  white-space: pre-wrap; }
    .score-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                   font-size: 0.85em; font-weight: 600; margin: 0 2px; }
    .score-high { background: #d1fae5; color: #065f46; }
    .score-mid  { background: #fef3c7; color: #92400e; }
    .score-low  { background: #fee2e2; color: #991b1b; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                    gap: 12px; margin: 16px 0; }
    .summary-card { background: white; border: 1px solid #e5e7eb; border-radius: 8px;
                    padding: 16px; text-align: center; }
    .summary-card .value { font-size: 1.8em; font-weight: 700; color: #7c3aed; }
    .summary-card .label { color: #666; font-size: 0.85em; }
</style>
"""


def _score_class(score: float) -> str:
    if score >= 4.0:
        return "score-high"
    elif score >= 3.0:
        return "score-mid"
    return "score-low"


def _find_winner(
    aggregates: dict[str, ModelAggregate], attr: str, lower_is_better: bool = False
) -> str:
    """Find the model with the best value for a given metric."""
    best_model = ""
    best_val = float("inf") if lower_is_better else float("-inf")
    for model, agg in aggregates.items():
        val = getattr(agg, attr)
        if lower_is_better:
            if val < best_val:
                best_val = val
                best_model = model
        else:
            if val > best_val:
                best_val = val
                best_model = model
    return best_model


def generate_summary_cards(aggregates: dict[str, ModelAggregate]) -> str:
    """Generate summary stat cards at the top of the report."""
    total_models = len(aggregates)
    total_cost = sum(a.total_cost_usd for a in aggregates.values())
    best_correct = _find_winner(aggregates, "avg_correctness")
    best_agg = aggregates.get(best_correct)
    best_score = f"{best_agg.avg_correctness:.2f}" if best_agg else "N/A"

    return f"""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{total_models}</div>
            <div class="label">Models Compared</div>
        </div>
        <div class="summary-card">
            <div class="value">{best_score}</div>
            <div class="label">Best Correctness ({html.escape(best_correct)})</div>
        </div>
        <div class="summary-card">
            <div class="value">${total_cost:.4f}</div>
            <div class="label">Total Cost</div>
        </div>
    </div>
    """


def generate_comparison_table(aggregates: dict[str, ModelAggregate]) -> str:
    """Generate the main HTML comparison table with winner highlighting."""
    winner_correct = _find_winner(aggregates, "avg_correctness")
    winner_style = _find_winner(aggregates, "avg_style")
    winner_complete = _find_winner(aggregates, "avg_completeness")
    winner_latency = _find_winner(
        aggregates, "avg_latency_seconds", lower_is_better=True
    )
    winner_cost = _find_winner(
        aggregates, "total_cost_usd", lower_is_better=True
    )

    rows = ""
    for model, agg in aggregates.items():
        safe_model = html.escape(model)

        def _cell(val: str, model_name: str, winner: str) -> str:
            cls = ' class="winner"' if model_name == winner else ""
            return f"<td{cls}>{val}</td>"

        rows += "<tr>"
        rows += f'<td class="model-name">{safe_model}</td>'
        rows += _cell(f"{agg.avg_correctness:.2f}", model, winner_correct)
        rows += _cell(f"{agg.avg_style:.2f}", model, winner_style)
        rows += _cell(f"{agg.avg_completeness:.2f}", model, winner_complete)
        rows += _cell(f"{agg.avg_latency_seconds:.2f}s", model, winner_latency)
        rows += _cell(f"${agg.total_cost_usd:.4f}", model, winner_cost)
        rows += "</tr>\n"

    return f"""
    <table class="eval-table">
        <tr>
            <th>Model</th>
            <th>Correctness<br><span class="metric-header">avg 1-5</span></th>
            <th>Style<br><span class="metric-header">avg 1-5</span></th>
            <th>Completeness<br><span class="metric-header">avg 1-5</span></th>
            <th>Avg Latency<br><span class="metric-header">seconds</span></th>
            <th>Total Cost<br><span class="metric-header">USD</span></th>
        </tr>
        {rows}
    </table>
    """


def generate_per_problem_breakdown(
    results: list[InferenceResult],
    evaluations: list[EvaluationResult],
) -> str:
    """Generate per-problem detail cards showing each model's output and scores."""
    # Group by task_id
    by_task: dict[str, list[tuple[InferenceResult, EvaluationResult | None]]] = {}
    eval_lookup = {
        (e.task_id, e.model): e for e in evaluations
    }
    for r in results:
        ev = eval_lookup.get((r.task_id, r.model))
        by_task.setdefault(r.task_id, []).append((r, ev))

    cards = ""
    for task_id in sorted(by_task.keys()):
        items = by_task[task_id]
        prompt_text = html.escape(items[0][0].prompt[:500])

        model_rows = ""
        for result, evaluation in items:
            safe_model = html.escape(result.model)
            safe_code = html.escape(result.generated_code[:1000])
            if evaluation:
                s = evaluation.score
                c_cls = _score_class(s.correctness)
                st_cls = _score_class(s.style)
                cp_cls = _score_class(s.completeness)
                scores_html = (
                    f'<span class="score-badge {c_cls}">C:{s.correctness}</span>'
                    f'<span class="score-badge {st_cls}">S:{s.style}</span>'
                    f'<span class="score-badge {cp_cls}">Cp:{s.completeness}</span>'
                )
                reasoning = html.escape(evaluation.score.reasoning[:300])
            else:
                scores_html = '<span class="score-badge score-low">No eval</span>'
                reasoning = ""

            model_rows += f"""
            <div style="margin: 8px 0; padding: 8px; border-left: 3px solid #7c3aed;">
                <strong>{safe_model}</strong> {scores_html}
                <div class="code-block">{safe_code}</div>
                {f'<p style="font-size: 0.85em; color: #666;"><em>{reasoning}</em></p>' if reasoning else ''}
            </div>
            """

        cards += f"""
        <div class="problem-card">
            <h3>{html.escape(task_id)}</h3>
            <div class="code-block">{prompt_text}</div>
            {model_rows}
        </div>
        """

    return cards


def build_full_report(report: CodeEvalReport) -> str:
    """Build the complete HTML report from a CodeEvalReport."""
    summary = generate_summary_cards(report.aggregates)
    table = generate_comparison_table(report.aggregates)
    breakdown = generate_per_problem_breakdown(
        report.results, report.evaluations
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(report.report_title)}</title>
    {REPORT_CSS}
</head>
<body>
    <h1>{html.escape(report.report_title)}</h1>
    <p class="subtitle">Generated {html.escape(report.generated_at)}
       &mdash; {len(report.test_case_ids)} problems &times; {len(report.models)} models</p>

    <div class="section">
        <h2>Summary</h2>
        {summary}
    </div>

    <div class="section">
        <h2>Model Comparison</h2>
        {table}
    </div>

    <div class="section">
        <h2>Per-Problem Breakdown</h2>
        {breakdown}
    </div>
</body>
</html>"""
