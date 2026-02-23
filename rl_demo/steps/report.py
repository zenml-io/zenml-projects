"""Create HTML visualization report: leaderboard table + training curves."""

import base64
import io
from typing import Annotated

from zenml import step
from zenml.types import HTMLString

from steps.models import EvalResult, TrainingResult


@step
def create_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> Annotated[HTMLString, "sweep_report"]:
    """
    Create an HTML visualization report: leaderboard table + training curves.

    Returns HTMLString for display in the ZenML dashboard.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        has_plt = True
    except ImportError:
        has_plt = False

    sorted_evals = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
    rows = []
    for r in sorted_evals:
        best_badge = " 🏆" if r.is_best else ""
        rows.append(
            f"<tr><td>{r.tag}</td><td>{r.eval_mean_reward:.2f} ± {r.eval_std_reward:.2f}</td>"
            f"<td>{r.eval_episodes}</td><td>{r.env_name}{best_badge}</td></tr>"
        )
    table_rows = "\n".join(rows)

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

    html = f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 900px;">
        <h2>RL Sweep Report</h2>
        <h3>Leaderboard</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background: #eee;">
                    <th style="padding: 0.5rem; text-align: left;">Tag</th>
                    <th style="padding: 0.5rem; text-align: left;">Eval Reward</th>
                    <th style="padding: 0.5rem; text-align: left;">Episodes</th>
                    <th style="padding: 0.5rem; text-align: left;">Environment</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        {curve_html}
    </div>
    """
    return HTMLString(html)
