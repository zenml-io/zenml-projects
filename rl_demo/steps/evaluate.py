"""Evaluate all trained policies and rank them."""

from typing import Annotated, Tuple

import wandb
from steps.experiment_tracking import active_wandb_tracker_name
from steps.helpers import make_policy, make_vecenv, run_eval_episodes
from steps.models import EvalResult, PolicyCheckpoint, TrainingResult
from zenml import log_metadata, step
from zenml.types import HTMLString


@step(experiment_tracker=active_wandb_tracker_name())
def evaluate_agents(
    training_results: list[TrainingResult],
    policy_checkpoints: list[PolicyCheckpoint],
    eval_episodes: int = 100,
) -> Tuple[
    Annotated[list[EvalResult], "eval_results"],
    Annotated[HTMLString, "leaderboard"],
]:
    """
    Evaluate all trained policies and rank them.

    Fan-in step: receives ALL training results and checkpoints as artifacts,
    runs evaluation episodes, and produces a ranked list.
    """
    print(f"📊 Evaluating {len(training_results)} trained agents...")
    assert len(training_results) == len(policy_checkpoints), (
        "results and checkpoints must be parallel lists"
    )

    eval_results = []
    for result, checkpoint in zip(training_results, policy_checkpoints):
        vecenv, _, _ = make_vecenv(
            result.env_name,
            vec={"num_envs": 1, "backend": "Serial"},
            env={"num_envs": 32},
        )
        policy = make_policy(vecenv, checkpoint=checkpoint)
        rewards = run_eval_episodes(policy, vecenv, eval_episodes)
        vecenv.close()

        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        std_r = (
            (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if rewards
            else 0.0
        )

        eval_results.append(
            EvalResult(
                env_name=result.env_name,
                tag=result.tag,
                eval_mean_reward=mean_r,
                eval_std_reward=std_r,
                eval_episodes=len(rewards),
                is_best=False,
            )
        )
        print(f"   {result.tag}: {mean_r:.2f} ± {std_r:.2f}")

    for env_name in set(r.env_name for r in eval_results):
        best = max(
            (r for r in eval_results if r.env_name == env_name),
            key=lambda r: r.eval_mean_reward,
        )
        best.is_best = True

    sorted_evals = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
    leaderboard_table = wandb.Table(
        columns=[
            "rank",
            "tag",
            "environment",
            "eval_mean_reward",
            "eval_std_reward",
            "eval_episodes",
            "is_best",
        ]
    )
    for rank, result in enumerate(sorted_evals, start=1):
        leaderboard_table.add_data(
            rank,
            result.tag,
            result.env_name,
            float(result.eval_mean_reward),
            float(result.eval_std_reward),
            int(result.eval_episodes),
            result.is_best,
        )
    best_overall = sorted_evals[0] if sorted_evals else None
    wandb_payload = {"eval/leaderboard": leaderboard_table}
    if best_overall:
        wandb_payload.update(
            {
                "eval/best_mean_reward": float(
                    best_overall.eval_mean_reward
                ),
                "eval/best_std_reward": float(best_overall.eval_std_reward),
            }
        )
        wandb.summary["best_eval_tag"] = best_overall.tag
        wandb.summary["best_eval_env"] = best_overall.env_name
    wandb.log(wandb_payload)

    log_metadata(
        metadata={
            "leaderboard": {
                r.tag: {
                    "reward": float(r.eval_mean_reward),
                    "std": float(r.eval_std_reward),
                }
                for r in sorted(
                    eval_results, key=lambda r: -r.eval_mean_reward
                )
            }
        },
        artifact_name="eval_results",
        infer_artifact=True,
    )

    # HTML leaderboard for ZenML dashboard
    rows = "".join(
        f"<tr><td>{r.tag}</td><td>{r.eval_mean_reward:.2f} ± {r.eval_std_reward:.2f}</td>"
        f"<td>{r.eval_episodes}</td><td>{r.env_name}{' 🏆' if r.is_best else ''}</td></tr>"
        for r in sorted_evals
    )
    leaderboard_html = HTMLString(f"""
    <div style="font-family: system-ui, sans-serif; padding: 1rem;">
        <h3>Evaluation Leaderboard</h3>
        <p><b>{len(eval_results)}</b> policies evaluated ({eval_episodes} episodes each)</p>
        <table style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background: #eee;">
                    <th style="padding: 0.5rem; text-align: left;">Tag</th>
                    <th style="padding: 0.5rem; text-align: left;">Eval Reward</th>
                    <th style="padding: 0.5rem; text-align: left;">Episodes</th>
                    <th style="padding: 0.5rem; text-align: left;">Environment</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """)
    return eval_results, leaderboard_html
