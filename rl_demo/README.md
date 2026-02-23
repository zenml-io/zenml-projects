# PufferLib × ZenML: Dynamic RL Training Pipeline

A demo showing how ZenML's dynamic pipelines orchestrate PufferLib RL training across multiple environments, with experiment tracking, artifact versioning, data lineage, and policy promotion — all in one pipeline.

## What This Pipeline Does

The pipeline trains reinforcement learning agents on multiple game environments (Pong, Breakout, Connect4, etc.), sweeps over hyperparameters (e.g. learning rates), evaluates all trained policies, and promotes the best-performing ones to production — with full traceability from model back to training data.

**In one run you get:**

- **N × M training runs** — N environments × M learning rates, each as an isolated ZenML step
- **Centralized evaluation** — All policies compared on the same criteria
- **HTML report** — Leaderboard table + training curves (mean reward, steps/sec)
- **Production promotion** — Best policy per environment promoted via ZenML Model Control Plane
- **Full lineage** — Every model version traces back to its dataset, client, and project (for GDPR/compliance)

## Pipeline Architecture

```
┌─────────────────────┐
│  load_training_data  │  ← Register dataset with client/project metadata
└──────────┬──────────┘     (root of lineage graph for compliance/scrubbing)
           │ DatasetMetadata
           ▼
┌─────────────────────┐
│  configure_sweep     │  ← Define envs + hyperparams at runtime
└──────────┬──────────┘
           │ list[EnvConfig]
           ▼
┌─────────────────────┐
│  train_agent         │  ← .map() fans out: one step per config
│  train_agent         │    Each runs in its own container, tracks
│  train_agent         │    artifacts, logs, metadata, retries
│  ...                 │
└──────────┬──────────┘
           │ list[TrainingResult]
           ▼
┌─────────────────────┐
│  evaluate_agents     │  ← Fan-in: compare all policies, rank them
└──────────┬──────────┘
           │ list[EvalResult]
           ▼
┌─────────────────────┐
│  create_sweep_report │  ← HTML: leaderboard + training curves
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  promote_best_policy │  ← Stage transition: staging → production
└─────────────────────┘     (Model Control Plane)
```

## Step-by-Step

| Step | Purpose |
|------|---------|
| **load_training_data** | Creates a versioned `DatasetMetadata` artifact with `client_id`, `project`, `data_source`, and `domain`. This is the **root of the lineage graph** — when legal says "delete all data for Client X", you can find every model version that touched that client's data. |
| **configure_sweep** | Produces the Cartesian product of `env_names × learning_rates`. The number of downstream training steps is determined **at runtime** (dynamic pipeline), not hardcoded. |
| **train_agent** | Uses PufferLib's PuffeRL trainer (PPO-style) to train an MLP policy on one environment. `.map()` creates one step per config; each runs in isolation with its own GPU. Returns `(TrainingResult, PolicyCheckpoint)` — checkpoints are dedicated artifacts, materialized and stored in the artifact store for cross-step access. |
| **evaluate_agents** | Receives `training_results` and `policy_checkpoints` as artifacts. Runs 100 evaluation episodes per policy, computes mean ± std reward, marks the best per environment. |
| **create_sweep_report** | Renders an HTML report: leaderboard table + matplotlib training curves (mean reward, steps/sec). Displayed in the ZenML dashboard. |
| **promote_best_policy** | Promotes the model version to `production` stage. Returns promoted checkpoints as **model artifacts** — load via `model_version.get_artifact("promoted_policy_checkpoints").load()` to get `dict[env_name, PolicyCheckpoint]`. |

## Key Concepts

### Checkpoints as dedicated artifacts

Checkpoints are **dedicated step outputs** (not paths inside `TrainingResult`). They flow as ZenML artifacts between steps and can be linked to the model in the promote step. This ensures:
- Cross-step access when steps run on different machines
- Proper lineage and versioning in the artifact store
- Model artifact linkage: `model_version.get_artifact("promoted_policy_checkpoints").load()` returns the winning policies.

### Dynamic pipeline

The sweep size is computed in `configure_sweep`, not in the pipeline definition. You could pull env names or hyperparams from a config file, database, or API — the pipeline adapts.

### Fan-out / fan-in

- **Fan-out**: `train_agent.map(configs)` creates one step per config (e.g. 6 steps for 3 envs × 2 LRs).
- **Fan-in**: `evaluate_agents` receives all results and compares them in one step.

### Model Control Plane

ZenML tracks a model `rl_policy` with versions. Each run creates a new version; `promote_best_policy` moves the latest to `production`. In the dashboard you see stage transitions (staging → production), linked artifacts, and full lineage.

### Data lineage & compliance

Every artifact is tagged with `client_id`, `project`, `data_source`, `domain`. Use ZenML's lineage to:

- Find all model versions trained on a given client's data
- Scrub or delete artifacts for a client (GDPR)
- Trace: promoted model → eval results → training run → dataset → client

## Requirements

```bash
# With pip
pip install -r requirements.txt

# With uv
uv pip install -r requirements.txt
```

## Running the Pipeline

```bash
# With your venv activated (direnv or manually)
python rl_demo.py

# Or with uv
uv run --active python rl_demo.py
```

### Before running

1. **ZenML server**: Start the ZenML server so you can use the dashboard and Model Control Plane:
   ```bash
   zenml up
   ```

2. **Connect**: Register a stack and connect to the server if needed:
   ```bash
   zenml connect
   ```

### Example runs

Default run (in `rl_demo.py`):

- 3 environments: `ocean-pong`, `ocean-breakout`, `ocean-connect4`
- 2 learning rates: `3e-4`, `1e-3`
- 100K timesteps each
- Device: cuda / mps (Apple Silicon) / cpu

Custom sweep:

```python
rl_environment_sweep(
    env_names=["ocean-snake"],
    learning_rates=[1e-4, 3e-4, 1e-3, 3e-3],
    total_timesteps=2_000_000,
    client_id="my-client",
    project="my-project",
)
```

## Infrastructure

The pipeline is infrastructure-agnostic. Switch stacks:

```bash
zenml stack set local          # Run on your laptop
zenml stack set k8s-gpu        # Fan out to Kubernetes pods with GPUs
zenml stack set slurm-cluster  # Submit to HPC job scheduler
```

The pipeline code stays the same; only the stack definition changes.

## Project Layout

```
rl_demo/
├── rl_demo.py      # Pipeline, steps, policy, helpers
├── requirements.txt
├── .envrc          # direnv: source .venv/bin/activate
└── README.md
```

## License

MIT
