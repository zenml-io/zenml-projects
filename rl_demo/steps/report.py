"""Create HTML visualization report: leaderboard table + training curves."""

from typing import Annotated

from steps.models import EvalResult, TrainingResult
from zenml import step
from zenml.types import HTMLString


def _sweep_summary_card(training_results: list[TrainingResult]) -> str:
    """Render the top-of-report summary: envs, # runs, lrs, timesteps."""
    if not training_results:
        return ""
    envs = sorted({r.env_name for r in training_results})
    lrs = sorted(
        {
            r.config.get("learning_rate")
            for r in training_results
            if r.config.get("learning_rate") is not None
        }
    )
    timesteps = sorted({r.total_timesteps for r in training_results})
    timesteps_str = ", ".join(f"{t:,}" for t in timesteps)
    return (
        f'<div style="background: #f7f7f9; padding: 0.75rem 1rem; '
        f'border-radius: 6px; margin: 0.5rem 0 1rem;">'
        f"<b>Sweep summary</b> &middot; "
        f"<b>{len(training_results)}</b> runs &middot; "
        f"envs: {', '.join(envs)} &middot; "
        f"learning rates: {', '.join(str(lr) for lr in lrs)} &middot; "
        f"timesteps per run: {timesteps_str}"
        f"</div>"
    )


def _headline_callout(eval_results: list[EvalResult]) -> str:
    """Render the headline that names the winner and delta vs runner-up."""
    if not eval_results:
        return ""
    ranked = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
    winner = ranked[0]
    if len(ranked) >= 2:
        delta = winner.eval_mean_reward - ranked[1].eval_mean_reward
        delta_str = (
            f' <span style="color: #2a7;">(+{delta:.2f} over runner-up)</span>'
        )
    else:
        delta_str = ""
    return (
        f'<p style="font-size: 1.1rem; margin: 0.5rem 0 1rem;">'
        f"🏆 <b>{winner.tag}</b> won eval: "
        f"<b>{winner.eval_mean_reward:.2f}</b> reward{delta_str}"
        f"</p>"
    )


def _leaderboard_table(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> str:
    """Render the enriched leaderboard table, sorted by eval reward desc."""
    train_by_tag = {t.tag: t for t in training_results}
    ranked = sorted(eval_results, key=lambda r: -r.eval_mean_reward)

    rows = []
    for rank, ev in enumerate(ranked, start=1):
        tr = train_by_tag.get(ev.tag)
        lr = tr.config.get("learning_rate", "—") if tr else "—"
        best_train = f"{tr.mean_reward:.2f}" if tr else "—"
        total_steps = f"{tr.total_timesteps:,}" if tr else "—"
        badge = " 🏆" if ev.is_best else ""
        row_style = ' style="background: #f3fbf3;"' if ev.is_best else ""
        rows.append(
            f"<tr{row_style}>"
            f"<td>{rank}</td>"
            f"<td>{ev.tag}{badge}</td>"
            f"<td>{lr}</td>"
            f"<td>{best_train}</td>"
            f"<td>{ev.eval_mean_reward:.2f} ± {ev.eval_std_reward:.2f}</td>"
            f"<td>{ev.eval_episodes}</td>"
            f"<td>{total_steps}</td>"
            f"<td>{ev.env_name}</td>"
            f"</tr>"
        )
    body = "\n".join(rows)
    return f"""
    <h3>Leaderboard</h3>
    <table style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr style="background: #eee;">
                <th style="padding: 0.5rem; text-align: left;">Rank</th>
                <th style="padding: 0.5rem; text-align: left;">Tag</th>
                <th style="padding: 0.5rem; text-align: left;">Learning rate</th>
                <th style="padding: 0.5rem; text-align: left;">Best train reward</th>
                <th style="padding: 0.5rem; text-align: left;">Eval reward</th>
                <th style="padding: 0.5rem; text-align: left;">Eval episodes</th>
                <th style="padding: 0.5rem; text-align: left;">Total steps</th>
                <th style="padding: 0.5rem; text-align: left;">Env</th>
            </tr>
        </thead>
        <tbody>
            {body}
        </tbody>
    </table>
    """


def _reward_curve(training_results: list[TrainingResult]) -> str:
    """Render the interactive reward-vs-iteration chart as a Plotly HTML fragment."""
    runs = [r for r in training_results if r.metrics_history]
    if not runs:
        return ""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    fig = go.Figure()
    for result in runs:
        hist = result.metrics_history
        iters = [h.get("iteration", i) for i, h in enumerate(hist)]
        rewards = [h.get("mean_reward", 0) for h in hist]
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=rewards,
                mode="lines+markers",
                name=result.tag,
                hovertemplate="iter %{x}<br>reward %{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        title="Training: Mean Reward",
        xaxis_title="Iteration",
        yaxis_title="Mean reward",
        height=400,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _sps_curve(training_results: list[TrainingResult]) -> str:
    """Render the interactive SPS-vs-iteration chart as a Plotly HTML fragment."""
    runs = [r for r in training_results if r.metrics_history]
    if not runs:
        return ""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return ""

    fig = go.Figure()
    for result in runs:
        hist = result.metrics_history
        iters = [h.get("iteration", i) for i, h in enumerate(hist)]
        sps = [h.get("sps", 0) for h in hist]
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=sps,
                mode="lines+markers",
                name=result.tag,
                hovertemplate="iter %{x}<br>sps %{y:.0f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        title="Throughput (Steps per Second)",
        xaxis_title="Iteration",
        yaxis_title="Steps/sec",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _render_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> str:
    """Pure renderer: returns the HTML body for the sweep report."""
    summary = _sweep_summary_card(training_results)
    headline = _headline_callout(eval_results)
    leaderboard = _leaderboard_table(training_results, eval_results)
    reward_chart = _reward_curve(training_results)
    chart_section = (
        f"<h3>Training: Mean Reward</h3>{reward_chart}" if reward_chart else ""
    )
    sps_chart = _sps_curve(training_results)
    sps_section = (
        f'<details style="margin-top: 1rem;">'
        f"<summary>Throughput (Steps per Second)</summary>"
        f"{sps_chart}</details>"
        if sps_chart
        else ""
    )
    return f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 1100px;">
        <h2>RL Sweep Report</h2>
        {summary}
        {headline}
        {leaderboard}
        {chart_section}
        {sps_section}
    </div>
    """


@step
def create_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> Annotated[HTMLString, "sweep_report"]:
    """HTML visualization report: leaderboard table + training curves."""
    return HTMLString(_render_sweep_report(training_results, eval_results))
