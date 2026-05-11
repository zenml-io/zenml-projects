# RL Sweep Report Redesign

**Status:** Approved
**Date:** 2026-05-11
**Scope:** `steps/report.py` (substantive rewrite), `requirements.txt` (deps change), one new unit test.

## Problem

The current end-of-pipeline `create_sweep_report` produces an HTML artifact that's hard to interpret:

- The leaderboard has only four columns (Tag, Eval Reward ± std, Episodes, Environment). There's no learning-rate column — to know which run is which, the reader has to parse the `tag` string (`{env_name}_lr{lr}`).
- The training curves are static matplotlib PNGs embedded as base64. No hover values, no way to toggle individual runs, hard to read when many runs overlap.
- No "headline" — the report doesn't tell the reader *which* run won, *by how much*, or summarize the sweep configuration.
- Throughput (SPS) and reward charts compete for visual weight even though reward is what matters.

## Goals

1. **Clearer narrative** — a reader who has never seen this pipeline can land on the report and immediately know: what was swept, which run won, by how much.
2. **Interactive charts** — hover for exact values, click legend entries to toggle runs on/off.

Out of scope (considered and explicitly dropped):

- Live mid-training progress in the dashboard.
- Surfacing additional metrics (policy/value loss, entropy, episode length) — already tracked but intentionally not displayed for now to keep the report focused on reward.
- A separate dashboard service (Streamlit/Gradio) — overkill for a demo pipeline.

## Design

### Layout (top to bottom)

1. **Sweep summary card** — environments, total number of runs, list of learning rates, total timesteps per run. All derived from `training_results[*].config`.
2. **Headline callout** — e.g. *"🏆 ocean-squared_lr0.05 won eval: 12.4 reward, +8.2 over runner-up"*. Computed from the top two entries of `eval_results` (sorted by `eval_mean_reward` desc). If only one run exists, the delta is omitted.
3. **Leaderboard table** — replaces the current 4-column table. Columns:
   `Rank | Tag | Learning rate | Best train reward | Eval reward ± std | Eval episodes | Total steps | Env`
   Sorted by `eval_mean_reward` desc. The winner row is tinted and shows 🏆. Learning rate is pulled from `result.config["learning_rate"]`, not parsed out of the tag string.
4. **Reward curve** — Plotly line chart. X: iteration. Y: mean_reward. One trace per run, hover tooltip shows `(iter, reward, tag)`, legend entries are click-toggleable. Embedded via `fig.to_html(include_plotlyjs="cdn", full_html=False)` so the HTML artifact stays a few KB rather than ballooning by ~3 MB.
5. **Throughput curve** — same chart shape, SPS vs iteration, wrapped in `<details>` so it's collapsed by default. Reward stays the focal point.

### Data flow

Inputs unchanged: `training_results: list[TrainingResult]`, `eval_results: list[EvalResult]`.

New derivations inside the step:

- `lr = result.config["learning_rate"]` — already present on `TrainingResult.config` (dump of `EnvConfig`).
- `best_train_reward = result.mean_reward` — the training loop in `steps/train.py` already stores best-so-far in this field.
- Sweep summary fields are aggregated from the configs embedded in `training_results`. No new pipeline inputs, no new `DatasetMetadata` pass-through.

### Dependencies

- **Add** `plotly` to `requirements.txt`.
- **Remove** `matplotlib` from `requirements.txt`. After this change it has no remaining caller in the repo.

### Edge cases

| Case | Behavior |
| --- | --- |
| `plotly` import fails | Render report without charts (summary card + headline + table still useful). Use a `try/except ImportError` guard mirroring the current matplotlib guard. |
| All `metrics_history` lists empty | Omit both curve sections entirely. |
| Single run in the sweep | Headline shows winner reward, no "+X over runner-up" delta. No rank-2 row. |
| CDN unreachable at view time | Chart `<div>` renders empty. Acceptable for a demo; documented as a known limitation. Can switch to `include_plotlyjs=True` later if a viewer needs offline rendering. |

### Testing

Add one unit test that calls `create_sweep_report` with two synthetic `TrainingResult` + `EvalResult` pairs and asserts the returned HTML contains:

- The leaderboard header row.
- Both run tags.
- A Plotly `<div>` (presence of a Plotly-generated `id` substring is sufficient).
- The 🏆 winner badge.

This is a smoke test against obvious regressions, not a visual diff. Full validation is by running the pipeline and viewing the dashboard.

## Non-goals / things deliberately left alone

- The per-run `training_summary` HTMLString produced by `train_agent` — stays as-is. We could embed a mini per-run curve there too, but it's out of scope.
- Pipeline structure, step signatures, materializers, model registration — unchanged.
- The `evaluate_agents` and `promote_best_policy` steps — unchanged.

## Risks

- **Plotly CDN dependency at view time.** Documented above. Acceptable for the demo.
- **Tag-to-learning-rate coupling** — we no longer parse the tag string, which is the right move, but downstream code that does parse `tag` is not affected by this change.
