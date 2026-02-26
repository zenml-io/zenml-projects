"""Evaluate all trained policies and rank them."""

from typing import Annotated, Tuple

from steps.helpers import make_policy, make_vecenv, run_eval_episodes
from steps.models import EvalResult, PolicyCheckpoint, TrainingResult
from zenml import log_metadata, step
from zenml.types import HTMLString


@step
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
    sorted_evals = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
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
