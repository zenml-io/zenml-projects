"""Tests for steps.report.create_sweep_report and its helpers."""

import pytest
from steps.models import EvalResult, TrainingResult
from steps.report import create_sweep_report


@pytest.fixture
def training_results():
    return [
        TrainingResult(
            env_name="ocean-squared",
            tag="ocean-squared_lr0.05",
            mean_reward=12.4,
            mean_episode_length=80.0,
            total_timesteps=1_000_000,
            steps_per_second=15000.0,
            policy_loss=0.1,
            value_loss=0.05,
            entropy=0.4,
            config={
                "learning_rate": 0.05,
                "env_name": "ocean-squared",
                "total_timesteps": 1_000_000,
            },
            metrics_history=[
                {"iteration": 0, "mean_reward": 1.0, "sps": 14000},
                {"iteration": 1, "mean_reward": 5.0, "sps": 15000},
                {"iteration": 2, "mean_reward": 10.0, "sps": 15000},
                {"iteration": 3, "mean_reward": 12.4, "sps": 15000},
            ],
        ),
        TrainingResult(
            env_name="ocean-squared",
            tag="ocean-squared_lr0.02",
            mean_reward=4.2,
            mean_episode_length=120.0,
            total_timesteps=1_000_000,
            steps_per_second=15500.0,
            policy_loss=0.15,
            value_loss=0.08,
            entropy=0.6,
            config={
                "learning_rate": 0.02,
                "env_name": "ocean-squared",
                "total_timesteps": 1_000_000,
            },
            metrics_history=[
                {"iteration": 0, "mean_reward": 0.5, "sps": 14500},
                {"iteration": 1, "mean_reward": 2.0, "sps": 15500},
                {"iteration": 2, "mean_reward": 3.5, "sps": 15500},
                {"iteration": 3, "mean_reward": 4.2, "sps": 15500},
            ],
        ),
    ]


@pytest.fixture
def eval_results():
    return [
        EvalResult(
            env_name="ocean-squared",
            tag="ocean-squared_lr0.05",
            eval_mean_reward=12.4,
            eval_std_reward=0.8,
            eval_episodes=20,
            is_best=True,
        ),
        EvalResult(
            env_name="ocean-squared",
            tag="ocean-squared_lr0.02",
            eval_mean_reward=4.2,
            eval_std_reward=0.5,
            eval_episodes=20,
            is_best=False,
        ),
    ]


def _render(training_results, eval_results) -> str:
    """Call the step's underlying function and return raw HTML."""
    return str(create_sweep_report.entrypoint(
        training_results=training_results, eval_results=eval_results
    ))


def test_baseline_contains_tags_and_leaderboard(training_results, eval_results):
    html = _render(training_results, eval_results)
    assert "Leaderboard" in html
    assert "ocean-squared_lr0.05" in html
    assert "ocean-squared_lr0.02" in html
