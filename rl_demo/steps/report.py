"""Create HTML visualization report: leaderboard table + training curves."""

import base64
import io
from typing import Annotated

from steps.models import EvalResult, TrainingResult
from zenml import step
from zenml.types import HTMLString


def _headline_callout(eval_results: list[EvalResult]) -> str:
    """Render the headline that names the winner and delta vs runner-up."""
    if not eval_results:
        return ""
    ranked = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
    winner = ranked[0]
    if len(ranked) >= 2:
        delta = winner.eval_mean_reward - ranked[1].eval_mean_reward
        delta_str = f" <span style=\"color: #2a7;\">(+{delta:.2f} over runner-up)</span>"
    else:
        delta_str = ""
    return (
        f"<p style=\"font-size: 1.1rem; margin: 0.5rem 0 1rem;\">"
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
        row_style = (
            ' style="background: #f3fbf3;"' if ev.is_best else ""
        )
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


def _render_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> str:
    """Pure renderer: returns the HTML body for the sweep report."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        has_plt = True
    except ImportError:
        has_plt = False

    curve_html = ""
    if has_plt and training_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for result in training_results:
            hist = result.metrics_history or []
            if not hist:
                continue
            iters = [h.get("iteration", i) for i, h in enumerate(hist)]
            rewards = [h.get("mean_reward", 0) for h in hist]
            sps = [h.get("sps", 0) for h in hist]
            axes[0].plot(iters, rewards, label=result.tag, alpha=0.8)
            axes[1].plot(iters, sps, label=result.tag, alpha=0.8)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean reward")
        axes[0].set_title("Training: Mean Reward")
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Steps/sec")
        axes[1].set_title("Training: Steps per Second")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        curve_html = f"""
        <h3>Training Curves</h3>
        <div style="margin: 1rem 0;">
            <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;">
        </div>
        """

    headline = _headline_callout(eval_results)
    leaderboard = _leaderboard_table(training_results, eval_results)
    return f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 900px;">
        <h2>RL Sweep Report</h2>
        {headline}
        {leaderboard}
        {curve_html}
    </div>
    """


@step
def create_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> Annotated[HTMLString, "sweep_report"]:
    """HTML visualization report: leaderboard table + training curves."""
    return HTMLString(_render_sweep_report(training_results, eval_results))
