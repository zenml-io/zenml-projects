"""Tests for steps.report.create_sweep_report and its helpers."""

import pytest
from steps.models import EvalResult, TrainingResult
from steps.report import (
    _headline_callout,
    _render_sweep_report,
    create_sweep_report,
)


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
    return str(
        create_sweep_report.entrypoint(
            training_results=training_results, eval_results=eval_results
        )
    )


def test_baseline_contains_tags_and_leaderboard(
    training_results, eval_results
):
    html = _render(training_results, eval_results)
    assert "Leaderboard" in html
    assert "ocean-squared_lr0.05" in html
    assert "ocean-squared_lr0.02" in html


def test_headline_shows_winner_and_delta(training_results, eval_results):
    html = _headline_callout(eval_results)
    assert "ocean-squared_lr0.05" in html
    assert "12.40" in html  # winner eval reward
    assert "+8.20" in html  # delta vs runner-up (12.4 - 4.2)


def test_headline_with_single_run_omits_delta():
    single = [
        EvalResult(
            env_name="ocean-squared",
            tag="only-run",
            eval_mean_reward=3.0,
            eval_std_reward=0.1,
            eval_episodes=10,
            is_best=True,
        )
    ]
    html = _headline_callout(single)
    assert "only-run" in html
    assert "3.00" in html
    assert "+" not in html  # no delta string


def test_headline_appears_in_full_report(training_results, eval_results):
    html = _render_sweep_report(training_results, eval_results)
    assert "+8.20" in html


from steps.report import _leaderboard_table


def test_leaderboard_columns(training_results, eval_results):
    html = _leaderboard_table(training_results, eval_results)
    # Header cells:
    for col in [
        "Rank",
        "Tag",
        "Learning rate",
        "Best train reward",
        "Eval reward",
        "Eval episodes",
        "Total steps",
        "Env",
    ]:
        assert col in html, f"missing column header: {col}"


def test_leaderboard_shows_learning_rate(training_results, eval_results):
    html = _leaderboard_table(training_results, eval_results)
    assert "0.05" in html
    assert "0.02" in html


def test_leaderboard_winner_row_has_badge(training_results, eval_results):
    html = _leaderboard_table(training_results, eval_results)
    assert "🏆" in html


def test_leaderboard_sorted_winner_first(training_results, eval_results):
    html = _leaderboard_table(training_results, eval_results)
    pos_winner = html.find("ocean-squared_lr0.05")
    pos_loser = html.find("ocean-squared_lr0.02")
    assert 0 <= pos_winner < pos_loser


from steps.report import _reward_curve


def test_reward_curve_uses_plotly(training_results):
    html = _reward_curve(training_results)
    # Plotly's to_html embeds either a "plotly-graph-div" class
    # or a call to "Plotly.newPlot" — either is sufficient evidence.
    assert "Plotly.newPlot" in html or "plotly-graph-div" in html


def test_reward_curve_includes_each_run_tag(training_results):
    html = _reward_curve(training_results)
    assert "ocean-squared_lr0.05" in html
    assert "ocean-squared_lr0.02" in html


def test_full_report_no_longer_uses_matplotlib_png(
    training_results, eval_results
):
    html = _render_sweep_report(training_results, eval_results)
    assert "data:image/png;base64" not in html


from steps.report import _sps_curve


def test_sps_curve_uses_plotly(training_results):
    html = _sps_curve(training_results)
    assert "Plotly.newPlot" in html or "plotly-graph-div" in html


def test_sps_chart_wrapped_in_collapsed_details(
    training_results, eval_results
):
    html = _render_sweep_report(training_results, eval_results)
    assert "<details" in html
    assert "Steps/sec" in html or "Steps per Second" in html


from steps.report import _sweep_summary_card


def test_sweep_summary_lists_envs_and_lrs(training_results):
    html = _sweep_summary_card(training_results)
    assert "ocean-squared" in html
    assert "0.05" in html
    assert "0.02" in html
    # Show the number of runs:
    assert "2" in html
    # Show the per-run total_timesteps in some human form:
    assert "1,000,000" in html or "1000000" in html


def test_sweep_summary_card_appears_in_full_report(
    training_results, eval_results
):
    html = _render_sweep_report(training_results, eval_results)
    assert "Sweep summary" in html


def test_empty_metrics_history_omits_charts(eval_results):
    bare = [
        TrainingResult(
            env_name="ocean-squared",
            tag="ocean-squared_lr0.05",
            mean_reward=0.0,
            mean_episode_length=0.0,
            total_timesteps=0,
            steps_per_second=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            config={"learning_rate": 0.05},
            metrics_history=[],
        ),
    ]
    html = _render_sweep_report(bare, eval_results[:1])
    assert "Plotly.newPlot" not in html
    assert "plotly-graph-div" not in html
    # Leaderboard + summary still render:
    assert "Leaderboard" in html
    assert "Sweep summary" in html


def test_single_run_no_delta_and_no_runner_up_row(
    training_results, eval_results
):
    html = _render_sweep_report(training_results[:1], eval_results[:1])
    assert "+" not in _headline_callout(eval_results[:1])
    # Only one tag in the leaderboard:
    assert html.count("ocean-squared_lr0.05") >= 1
    assert "ocean-squared_lr0.02" not in html


def test_plotly_missing_falls_back_gracefully(
    monkeypatch, training_results, eval_results
):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("plotly"):
            raise ImportError("plotly disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    html = _render_sweep_report(training_results, eval_results)
    assert "Plotly.newPlot" not in html
    assert "plotly-graph-div" not in html
    # Table + summary still render:
    assert "Leaderboard" in html
    assert "Sweep summary" in html
