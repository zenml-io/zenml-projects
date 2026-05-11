# RL Sweep Report Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static matplotlib-based RL sweep report with a Plotly-driven interactive report that has a clearer narrative (sweep summary card, headline callout, enriched leaderboard) and click-toggleable reward/SPS curves.

**Architecture:** Rewrite `steps/report.py`. Keep the `@step` decorator on `create_sweep_report` (signature unchanged). Extract a pure helper `_render_sweep_report(training_results, eval_results) -> str` and a set of small section helpers (`_sweep_summary_card`, `_headline_callout`, `_leaderboard_table`, `_reward_curve`, `_sps_curve`). Each section helper returns an HTML fragment so it can be unit-tested in isolation. The step just wraps the rendered string in `HTMLString`.

**Tech Stack:** ZenML 0.x (`@step`, `HTMLString`), Pydantic v2 (existing `TrainingResult`/`EvalResult`), Plotly (new), pytest (new dev dependency, install once).

**Spec:** `docs/superpowers/specs/2026-05-11-rl-sweep-report-redesign.md`

## File Structure

| File | Action | Responsibility |
| --- | --- | --- |
| `steps/report.py` | Rewrite body | Step entrypoint + pure renderer + section helpers |
| `tests/__init__.py` | Create (empty) | Marks `tests/` as a package so imports work cleanly |
| `tests/test_report.py` | Create | Fixtures + unit tests for the renderer and each section |
| `requirements.txt` | Modify | Add `plotly`, remove `matplotlib` |

All paths below are relative to `/Users/alexej/Repos/zenml-projects/rl_demo/`. Run commands from that directory.

---

### Task 1: Test scaffolding + baseline regression test

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Ensure pytest is installed**

Run: `python -c "import pytest; print(pytest.__version__)"`

If this raises `ModuleNotFoundError`, run: `pip install pytest`. Otherwise proceed.

- [ ] **Step 2: Create empty `tests/__init__.py`**

Create `tests/__init__.py` with zero bytes (touch it).

- [ ] **Step 3: Write fixtures and a baseline test against the existing report**

Create `tests/test_report.py` with this content:

```python
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
```

- [ ] **Step 4: Run the baseline test to confirm it passes against current code**

Run: `pytest tests/test_report.py::test_baseline_contains_tags_and_leaderboard -v`

Expected: PASS. (The current `report.py` already renders a Leaderboard heading and includes both tags in the table rows.)

If it fails with `ImportError: No module named 'steps'`, run pytest with the project as the import root: `PYTHONPATH=. pytest tests/test_report.py -v`. If that works, add a `conftest.py` at the repo root:

```python
# conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

Re-run pytest without `PYTHONPATH`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_report.py
# Include conftest.py if created in step 4:
[ -f conftest.py ] && git add conftest.py
git commit -m "test: scaffolding + baseline test for create_sweep_report"
```

---

### Task 2: Add Plotly dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add plotly to requirements.txt**

Edit `requirements.txt` so it reads:

```
zenml[server]
pufferlib
torch
matplotlib
plotly
numpy<3.0
psutil
```

(We keep `matplotlib` for now; it gets removed in Task 10 after the chart code no longer needs it.)

- [ ] **Step 2: Install plotly locally**

Run: `pip install plotly`

Expected: installation succeeds. Verify with `python -c "import plotly; print(plotly.__version__)"` — should print a version, no error.

- [ ] **Step 3: Re-run baseline test**

Run: `pytest tests/test_report.py -v`

Expected: still PASS. No code touched yet.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "deps: add plotly for interactive sweep report"
```

---

### Task 3: Extract pure renderer function

**Files:**
- Modify: `steps/report.py`

This is a non-behavior-changing refactor that introduces the helper boundary so subsequent TDD tasks can test pieces in isolation.

- [ ] **Step 1: Refactor `steps/report.py` to introduce `_render_sweep_report`**

Replace the entire contents of `steps/report.py` with:

```python
"""Create HTML visualization report: leaderboard table + training curves."""

import base64
import io
from typing import Annotated

from steps.models import EvalResult, TrainingResult
from zenml import step
from zenml.types import HTMLString


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

    return f"""
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


@step
def create_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> Annotated[HTMLString, "sweep_report"]:
    """HTML visualization report: leaderboard table + training curves."""
    return HTMLString(_render_sweep_report(training_results, eval_results))
```

- [ ] **Step 2: Run the baseline test to confirm refactor is behavior-preserving**

Run: `pytest tests/test_report.py -v`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add steps/report.py
git commit -m "refactor: extract pure _render_sweep_report helper"
```

---

### Task 4: Headline callout

**Files:**
- Modify: `steps/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add failing tests for the headline**

Append to `tests/test_report.py`:

```python
from steps.report import _headline_callout, _render_sweep_report


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
```

- [ ] **Step 2: Run new tests to confirm they fail**

Run: `pytest tests/test_report.py -v`

Expected: 3 FAILs (ImportError on `_headline_callout`, then the third fails on the missing delta string in full report).

- [ ] **Step 3: Implement `_headline_callout` and wire it into the renderer**

In `steps/report.py`, add this helper above `_render_sweep_report`:

```python
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
```

Then in `_render_sweep_report`, inject the headline immediately after the `<h2>RL Sweep Report</h2>` line. Change the bottom return statement so it becomes:

```python
    headline = _headline_callout(eval_results)
    return f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 900px;">
        <h2>RL Sweep Report</h2>
        {headline}
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
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add steps/report.py tests/test_report.py
git commit -m "feat: headline callout naming winner and delta vs runner-up"
```

---

### Task 5: Enriched leaderboard

**Files:**
- Modify: `steps/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add failing tests for the leaderboard**

Append to `tests/test_report.py`:

```python
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
```

- [ ] **Step 2: Run to confirm failures**

Run: `pytest tests/test_report.py -v`

Expected: 4 FAILs (ImportError on `_leaderboard_table`).

- [ ] **Step 3: Implement `_leaderboard_table` and wire it in**

In `steps/report.py`, add this helper above `_render_sweep_report`:

```python
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
```

Then in `_render_sweep_report`, **delete** the old inline `<h3>Leaderboard</h3> ... </table>` block and the `rows`/`table_rows` construction. Replace with a single call. The function body becomes:

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS. The earlier baseline test (`Leaderboard` heading + both tags present) still holds because both are still rendered.

- [ ] **Step 5: Commit**

```bash
git add steps/report.py tests/test_report.py
git commit -m "feat: enriched leaderboard with rank, lr, best train reward, total steps"
```

---

### Task 6: Replace matplotlib reward chart with Plotly

**Files:**
- Modify: `steps/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add failing tests for the Plotly reward curve**

Append to `tests/test_report.py`:

```python
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


def test_full_report_no_longer_uses_matplotlib_png(training_results, eval_results):
    html = _render_sweep_report(training_results, eval_results)
    assert "data:image/png;base64" not in html
```

- [ ] **Step 2: Run to confirm failures**

Run: `pytest tests/test_report.py -v`

Expected: 3 FAILs (ImportError on `_reward_curve`, then the matplotlib-PNG test fails because the function still emits a base64 PNG).

- [ ] **Step 3: Implement `_reward_curve` and remove the matplotlib chart**

In `steps/report.py`:

1. **Add the helper** above `_render_sweep_report`:

```python
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
```

2. **Remove** the entire `try: import matplotlib ... has_plt = False` block, the `if has_plt and training_results: ...` block that builds `axes`/`buf`/`img_b64`, and the `curve_html = f"""<h3>Training Curves</h3>...` assignment.

3. **Remove** the now-unused imports at the top: `import base64`, `import io`.

4. **Update** the bottom return statement to call `_reward_curve` instead of using `curve_html`:

```python
    headline = _headline_callout(eval_results)
    leaderboard = _leaderboard_table(training_results, eval_results)
    reward_chart = _reward_curve(training_results)
    chart_section = (
        f"<h3>Training: Mean Reward</h3>{reward_chart}" if reward_chart else ""
    )
    return f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 1100px;">
        <h2>RL Sweep Report</h2>
        {headline}
        {leaderboard}
        {chart_section}
    </div>
    """
```

(The Plotly figure already has its own title; the extra `<h3>` is the section anchor in the surrounding HTML, kept for consistency with the leaderboard section.)

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add steps/report.py tests/test_report.py
git commit -m "feat: replace matplotlib chart with interactive Plotly reward curve"
```

---

### Task 7: Collapsed SPS chart

**Files:**
- Modify: `steps/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_report.py`:

```python
from steps.report import _sps_curve


def test_sps_curve_uses_plotly(training_results):
    html = _sps_curve(training_results)
    assert "Plotly.newPlot" in html or "plotly-graph-div" in html


def test_sps_chart_wrapped_in_collapsed_details(training_results, eval_results):
    html = _render_sweep_report(training_results, eval_results)
    assert "<details" in html
    assert "Steps/sec" in html or "Steps per Second" in html
```

- [ ] **Step 2: Run to confirm failures**

Run: `pytest tests/test_report.py -v`

Expected: 2 FAILs (ImportError on `_sps_curve`, no `<details>` in output).

- [ ] **Step 3: Implement `_sps_curve` and wire it into the renderer**

In `steps/report.py`, add this helper above `_render_sweep_report`:

```python
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
```

In the renderer, append a collapsed `<details>` block for the SPS chart. Update the bottom return so it includes:

```python
    sps_chart = _sps_curve(training_results)
    sps_section = (
        f"<details style=\"margin-top: 1rem;\">"
        f"<summary>Throughput (Steps per Second)</summary>"
        f"{sps_chart}</details>"
        if sps_chart
        else ""
    )
    return f"""
    <div style="font-family: system-ui, sans-serif; padding: 1.5rem; max-width: 1100px;">
        <h2>RL Sweep Report</h2>
        {headline}
        {leaderboard}
        {chart_section}
        {sps_section}
    </div>
    """
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add steps/report.py tests/test_report.py
git commit -m "feat: collapsed SPS chart in <details> beneath the reward curve"
```

---

### Task 8: Sweep summary card

**Files:**
- Modify: `steps/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_report.py`:

```python
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


def test_sweep_summary_card_appears_in_full_report(training_results, eval_results):
    html = _render_sweep_report(training_results, eval_results)
    assert "Sweep summary" in html
```

- [ ] **Step 2: Run to confirm failures**

Run: `pytest tests/test_report.py -v`

Expected: 2 FAILs.

- [ ] **Step 3: Implement `_sweep_summary_card` and inject it**

In `steps/report.py`, add above `_render_sweep_report`:

```python
def _sweep_summary_card(training_results: list[TrainingResult]) -> str:
    """Render the top-of-report summary: envs, # runs, lrs, timesteps."""
    if not training_results:
        return ""
    envs = sorted({r.env_name for r in training_results})
    lrs = sorted({r.config.get("learning_rate") for r in training_results
                  if r.config.get("learning_rate") is not None})
    timesteps = sorted({r.total_timesteps for r in training_results})
    timesteps_str = ", ".join(f"{t:,}" for t in timesteps)
    return (
        f"<div style=\"background: #f7f7f9; padding: 0.75rem 1rem; "
        f"border-radius: 6px; margin: 0.5rem 0 1rem;\">"
        f"<b>Sweep summary</b> &middot; "
        f"<b>{len(training_results)}</b> runs &middot; "
        f"envs: {', '.join(envs)} &middot; "
        f"learning rates: {', '.join(str(lr) for lr in lrs)} &middot; "
        f"timesteps per run: {timesteps_str}"
        f"</div>"
    )
```

In `_render_sweep_report`, inject the summary card immediately before the headline:

```python
    summary = _sweep_summary_card(training_results)
    headline = _headline_callout(eval_results)
    leaderboard = _leaderboard_table(training_results, eval_results)
    reward_chart = _reward_curve(training_results)
    chart_section = (
        f"<h3>Training: Mean Reward</h3>{reward_chart}" if reward_chart else ""
    )
    sps_chart = _sps_curve(training_results)
    sps_section = (
        f"<details style=\"margin-top: 1rem;\">"
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
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add steps/report.py tests/test_report.py
git commit -m "feat: sweep summary card showing envs, learning rates, timesteps"
```

---

### Task 9: Edge case handling

**Files:**
- Modify: `tests/test_report.py`
- Modify: `steps/report.py` (only if any test fails)

The renderer already handles most edge cases via the `if not training_results` / `if not runs` / `try-except ImportError` guards added in earlier tasks. This task is a verification pass.

- [ ] **Step 1: Add edge-case tests**

Append to `tests/test_report.py`:

```python
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


def test_single_run_no_delta_and_no_runner_up_row(training_results, eval_results):
    html = _render_sweep_report(training_results[:1], eval_results[:1])
    assert "+" not in _headline_callout(eval_results[:1])
    # Only one tag in the leaderboard:
    assert html.count("ocean-squared_lr0.05") >= 1
    assert "ocean-squared_lr0.02" not in html


def test_plotly_missing_falls_back_gracefully(monkeypatch, training_results, eval_results):
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
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS. If any fail, the failure is a real bug in the guards introduced in Tasks 4–8 — read the failure message and adjust the corresponding `if not ...` / `try-except ImportError` guard.

- [ ] **Step 3: Commit**

```bash
git add tests/test_report.py
git commit -m "test: edge cases (empty history, single run, plotly missing)"
```

---

### Task 10: Remove matplotlib + visual smoke check

**Files:**
- Modify: `requirements.txt`
- No code changes expected here.

- [ ] **Step 1: Confirm matplotlib has no remaining import in the repo**

Run: `grep -rn "matplotlib" --include='*.py' .`

Expected: zero matches. (The earlier rewrite of `steps/report.py` already removed both `import matplotlib` calls.)

If any match remains, stop and investigate before continuing.

- [ ] **Step 2: Remove matplotlib from requirements.txt**

Edit `requirements.txt` so it reads:

```
zenml[server]
pufferlib
torch
plotly
numpy<3.0
psutil
```

- [ ] **Step 3: Re-run the full test suite**

Run: `pytest tests/test_report.py -v`

Expected: ALL PASS.

- [ ] **Step 4: Render the report from fixtures and eyeball it**

Run from the `rl_demo/` directory:

```bash
python -c "
from tests.test_report import training_results, eval_results
from steps.report import _render_sweep_report
# pytest fixtures are functions returning data — call them via the fixture body:
import inspect
tr = training_results.__wrapped__() if hasattr(training_results, '__wrapped__') else None
ev = eval_results.__wrapped__() if hasattr(eval_results, '__wrapped__') else None
"
```

This is tricky because pytest fixtures aren't trivially callable. Instead, use a small inline script:

```bash
python - <<'PY'
from steps.models import EvalResult, TrainingResult
from steps.report import _render_sweep_report

tr = [
    TrainingResult(
        env_name="ocean-squared", tag="ocean-squared_lr0.05",
        mean_reward=12.4, mean_episode_length=80.0, total_timesteps=1_000_000,
        steps_per_second=15000.0, policy_loss=0.1, value_loss=0.05, entropy=0.4,
        config={"learning_rate": 0.05, "env_name": "ocean-squared", "total_timesteps": 1_000_000},
        metrics_history=[
            {"iteration": i, "mean_reward": r, "sps": 15000}
            for i, r in enumerate([1.0, 5.0, 10.0, 12.4])
        ],
    ),
    TrainingResult(
        env_name="ocean-squared", tag="ocean-squared_lr0.02",
        mean_reward=4.2, mean_episode_length=120.0, total_timesteps=1_000_000,
        steps_per_second=15500.0, policy_loss=0.15, value_loss=0.08, entropy=0.6,
        config={"learning_rate": 0.02, "env_name": "ocean-squared", "total_timesteps": 1_000_000},
        metrics_history=[
            {"iteration": i, "mean_reward": r, "sps": 15500}
            for i, r in enumerate([0.5, 2.0, 3.5, 4.2])
        ],
    ),
]
ev = [
    EvalResult(env_name="ocean-squared", tag="ocean-squared_lr0.05",
               eval_mean_reward=12.4, eval_std_reward=0.8, eval_episodes=20, is_best=True),
    EvalResult(env_name="ocean-squared", tag="ocean-squared_lr0.02",
               eval_mean_reward=4.2, eval_std_reward=0.5, eval_episodes=20, is_best=False),
]
html = _render_sweep_report(tr, ev)
open("/tmp/sweep_report_preview.html", "w").write(
    "<html><head><meta charset='utf-8'></head><body>" + html + "</body></html>"
)
print("Wrote /tmp/sweep_report_preview.html")
PY
```

Then open `/tmp/sweep_report_preview.html` in a browser and verify visually:

- Sweep summary card at top with envs, # runs, learning rates, timesteps.
- Headline says winner tag and `+8.20` over runner-up.
- Leaderboard table has all 8 columns and the winner row is tinted.
- Reward curve is interactive: hover shows values, click legend entries toggles traces.
- "Throughput (Steps per Second)" is collapsed (`<details>`) — clicking it expands the SPS chart.

If anything looks broken, fix the relevant section helper and re-run pytest before continuing.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "deps: drop matplotlib (no longer used after Plotly migration)"
```

---

## Self-Review (done by the planner)

**Spec coverage:**
- Sweep summary card → Task 8 ✓
- Headline callout → Task 4 ✓
- Enriched leaderboard (8 columns, sort, winner tint, lr from `config`) → Task 5 ✓
- Plotly reward curve (CDN-hosted, hover, click-toggle) → Task 6 ✓
- Collapsed SPS curve → Task 7 ✓
- Edge cases (empty history, single run, plotly missing) → Task 9 ✓
- Add plotly / remove matplotlib → Tasks 2 + 10 ✓
- Unit test smoke coverage → Tasks 1, 4–9 ✓
- Visual eyeball → Task 10 step 4 ✓

**Placeholder scan:** no TBDs, no "implement appropriately", every code step has full code, every command has expected output.

**Type consistency:** all section helpers consistently take `list[TrainingResult]` / `list[EvalResult]` and return `str`. `_render_sweep_report` returns `str`; the `@step` wraps in `HTMLString`. Test helper `_render` calls `create_sweep_report.entrypoint(...)` consistently throughout.
