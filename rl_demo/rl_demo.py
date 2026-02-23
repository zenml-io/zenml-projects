"""
PufferLib x ZenML: Dynamic RL Training Pipeline
====================================================

A demo showing how ZenML's dynamic pipelines orchestrate PufferLib RL
training across multiple environments, with automatic experiment tracking,
artifact versioning, data lineage, and policy promotion — all in one pipeline.

This is what "MLOps for RL" looks like when you don't want to pay 100K
for experiment tracking alone.

Requirements:
    pip install "zenml[server]" pufferlib torch

Architecture:
    ┌─────────────────┐
    │ load_training_data│  ← Register dataset with client/project metadata
    └────────┬────────┘     (root of lineage graph for compliance/scrubbing)
             │ DatasetMetadata
             ▼
    ┌─────────────────┐
    │  configure_sweep │  ← Define envs + hyperparams at runtime
    └────────┬────────┘
             │ list[EnvConfig]
             ▼
    ┌─────────────────┐
    │  train_agent     │  ← .map() fans out: one step per env config
    │  train_agent     │    Each is a tracked ZenML step with its own
    │  train_agent     │    artifacts, logs, metadata, and retries
    │  train_agent     │
    └────────┬────────┘
             │ list[TrainingResult]
             ▼
    ┌─────────────────┐
    │  evaluate_agents │  ← Fan-in: compare all policies
    └────────┬────────┘
             │ list[EvalResult]
             ▼
    ┌─────────────────┐
    │  promote_best    │  ← Stage transition: staging → production
    └─────────────────┘     (Model Control Plane with full lineage)
"""

import sys
import tempfile
import threading
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel
import torch
import torch.nn as nn

# ─── ZenML imports ───────────────────────────────────────────────────
from zenml import Model, pipeline, step, log_metadata
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.types import HTMLString

# ─── PufferLib imports ───────────────────────────────────────────────
from pufferlib.pufferl import PuffeRL
import pufferlib.pufferl as pufferl
from pufferlib.pytorch import layer_init

# ─── Docker / K8s settings ───────────────────────────────────────────
from zenml.config import DockerSettings
from zenml.config.docker_settings import DockerBuildConfig, DockerBuildOptions
from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import KubernetesOrchestratorSettings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA CLASSES — Pydantic models (avoids ZenML/Pydantic dataclass Field conflict)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EnvConfig(BaseModel):
    """Configuration for a single RL environment + hyperparameter combo."""
    env_name: str
    num_envs: int = 64
    num_workers: int = 2
    learning_rate: float = 3e-4
    batch_size: int = 8192
    total_timesteps: int = 100_000
    update_epochs: int = 3
    device: str = "cuda"
    tag: str = ""


class TrainingResult(BaseModel):
    """Output of a single training run — versioned as a ZenML artifact."""
    env_name: str
    tag: str
    mean_reward: float
    mean_episode_length: float
    total_timesteps: int
    steps_per_second: float
    policy_loss: float
    value_loss: float
    entropy: float
    checkpoint_path: str
    config: dict
    metrics_history: list[dict] = []


class EvalResult(BaseModel):
    """Evaluation result for a trained policy."""
    env_name: str
    tag: str
    eval_mean_reward: float
    eval_std_reward: float
    eval_episodes: int
    checkpoint_path: str
    is_best: bool = False


class DatasetMetadata(BaseModel):
    """
    Metadata for a versioned training dataset.

    Every training run traces back to this artifact — so when legal says
    "delete all data for Client X", you can find every model version
    that touched that data via the ZenML lineage graph.
    """
    client_id: str
    project: str
    data_source: str
    domain: str
    env_names: list[str]
    description: str = ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# POLICY — Standard PyTorch, no wrapper lock-in
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RLPolicy(nn.Module):
    """
    Simple MLP policy compatible with PufferLib's PuffeRL trainer.
    Swap this out for CNN, LSTM, Transformer — whatever your research needs.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(hidden, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, observations, state=None):
        features = self.encoder(observations)
        return self.actor(features), self.critic(features)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS — PufferLib env setup (shared by train + eval steps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _puffer_env_name(name: str) -> str:
    """Map 'ocean-pong' → 'puffer_pong' for PufferLib config lookup."""
    if name.startswith("ocean-"):
        return "puffer_" + name[6:]
    return name


_argv_lock = threading.Lock()


def _make_vecenv(env_name: str, **overrides):
    """Create a PufferLib vectorized env with config overrides."""
    puffer_name = _puffer_env_name(env_name)
    # PufferLib's load_config calls argparse.parse_args() which reads sys.argv.
    # In ZenML K8s pods, sys.argv contains ZenML entrypoint args that PufferLib
    # doesn't understand. Lock + clear argv so concurrent threads (from
    # dynamic pipeline's thread pool) don't race on sys.argv.
    with _argv_lock:
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            args = pufferl.load_config(puffer_name)
        finally:
            sys.argv = original_argv
    for section, updates in overrides.items():
        args[section].update(updates)
    return pufferl.load_env(puffer_name, args), puffer_name, args


def _make_policy(vecenv, device: str = "cpu", checkpoint_path: str = None):
    """Create an RLPolicy from vecenv spaces, optionally loading a checkpoint."""
    obs_dim = vecenv.single_observation_space.shape[0]
    act_dim = vecenv.single_action_space.n
    policy = RLPolicy(obs_dim, act_dim)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
    else:
        policy = policy.to(device)
    return policy


def _extract_logs(logs: dict | None) -> dict:
    """Pull standard metrics from PuffeRL log dict."""
    if logs is None:
        return {"mean_reward": 0, "mean_episode_length": 0,
                "policy_loss": 0, "value_loss": 0, "entropy": 0, "sps": 0}

    def _get(key, alt=None):
        return logs.get(key, logs.get(alt or key, 0))

    return {
        "mean_reward": _get("environment/mean_reward", "environment/score"),
        "mean_episode_length": _get("environment/mean_episode_length", "environment/episode_length"),
        "policy_loss": _get("losses/policy_loss"),
        "value_loss": _get("losses/value_loss"),
        "entropy": _get("losses/entropy"),
        "sps": _get("SPS"),
    }


def _run_eval_episodes(policy, vecenv, num_episodes: int) -> list[float]:
    """Run evaluation episodes and return per-episode rewards."""
    rewards = []
    obs, _ = vecenv.reset()
    episode_rewards = [0.0] * vecenv.num_agents
    completed = 0
    while completed < num_episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(torch.FloatTensor(obs))
            actions = logits.argmax(dim=-1).numpy()
        obs, reward, terminated, truncated, _ = vecenv.step(actions)
        for i in range(vecenv.num_agents):
            episode_rewards[i] += float(reward[i])
            if terminated[i] or truncated[i]:
                rewards.append(episode_rewards[i])
                episode_rewards[i] = 0.0
                completed += 1
    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS — Each is independently tracked, cached, and retriable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@step
def load_training_data(
    env_names: list[str],
    client_id: str = "acme-corp",
    project: str = "rl-optimization",
    data_source: str = "internal-simulation",
    domain: str = "operations-research",
) -> Annotated[DatasetMetadata, "training_dataset"]:
    """
    Register and version the training dataset with lineage metadata.

    This step creates a traceable artifact at the root of every training run.
    In the ZenML dashboard, you can click on any model version and trace back
    to this exact dataset — including which client, project, and data source
    produced it.

    For data scrubbing / GDPR compliance:
        zenml artifact list --tag client_id:acme-corp
        → shows every artifact (datasets, checkpoints, models) linked to that client
        → trace forward to find all model versions that need to be retrained or deleted
    """
    metadata = DatasetMetadata(
        client_id=client_id,
        project=project,
        data_source=data_source,
        domain=domain,
        env_names=env_names,
        description=f"Training data for {len(env_names)} environments, "
                    f"client={client_id}, project={project}",
    )

    # Attach rich metadata to the artifact — searchable and filterable in the dashboard
    log_metadata(
        metadata={
            "client_id": client_id,
            "project": project,
            "data_source": data_source,
            "domain": domain,
            "environments": env_names,
            "compliance": {
                "data_retention_policy": "90d",
                "scrub_eligible": True,
                "lineage_tracked": True,
            },
        },
        infer_artifact=True,
    )

    print(f"📦 Registered training dataset: client={client_id}, project={project}")
    print(f"   Data source: {data_source} | Domain: {domain}")
    print(f"   Environments: {env_names}")
    print(f"   → Full lineage tracked — traceable for data scrubbing/compliance")

    return metadata


@step
def configure_sweep(
    dataset: DatasetMetadata,
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
) -> Annotated[list[EnvConfig], "sweep_configs"]:
    """
    Generate the sweep configuration at runtime.

    This is the "dynamic" part — the number of downstream training steps
    is determined HERE, not hardcoded in the pipeline definition.
    You could also pull this from a config file, database, or API.
    """
    configs = []
    for env_name in dataset.env_names:
        for lr in learning_rates:
            tag = f"{env_name}_lr{lr}"
            configs.append(
                EnvConfig(
                    env_name=env_name,
                    learning_rate=lr,
                    total_timesteps=total_timesteps,
                    device=device,
                    tag=tag,
                )
            )

    # Log sweep metadata — infer_artifact attaches to configure_sweep output
    log_metadata(
        metadata={
            "sweep_size": len(configs),
            "environments": dataset.env_names,
            "learning_rates": learning_rates,
            "client_id": dataset.client_id,
            "project": dataset.project,
        },
        infer_artifact=True,
    )

    print(f"🐡 Configured sweep: {len(configs)} training runs")
    for c in configs:
        print(f"   → {c.tag}")

    return configs


@step
def train_agent(
    config: EnvConfig,
) -> Annotated[TrainingResult, "training_result"]:
    """
    Train a single RL agent using PufferLib's PuffeRL trainer.

    This step runs independently per environment config. ZenML handles:
    - Containerization (same Docker image, different params)
    - GPU allocation via the orchestrator (K8s, Slurm, local)
    - Artifact versioning of the trained checkpoint
    - Automatic retry on failure
    """
    print(f"🎮 Training on {config.env_name} | lr={config.learning_rate}")

    backend = "Serial" if config.num_workers <= 1 else "Multiprocessing"
    vec_overrides = {"num_envs": max(1, config.num_workers), "backend": backend}
    if backend == "Multiprocessing":
        vec_overrides["num_workers"] = config.num_workers

    vecenv, puffer_name, args = _make_vecenv(
        config.env_name,
        vec=vec_overrides,
        env={"num_envs": config.num_envs},
        train={
            "device": config.device,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "total_timesteps": config.total_timesteps,
            "update_epochs": config.update_epochs,
            "optimizer": "adam",
            "use_rnn": False,
            "minibatch_size": min(4096, config.batch_size),
            "max_minibatch_size": config.batch_size,
        },
    )
    policy = _make_policy(vecenv, device=config.device)
    trainer = PuffeRL(dict(**args["train"], env=puffer_name), vecenv, policy)

    # ── Training loop ─────────────────────────────────────────────
    best_reward = float("-inf")
    metrics_history = []

    while trainer.global_step < config.total_timesteps:
        trainer.evaluate()
        logs = trainer.train()

        stats = _extract_logs(logs)
        if stats["mean_reward"] > best_reward:
            best_reward = stats["mean_reward"]

        if logs is not None:
            metrics_history.append({"iteration": len(metrics_history), **stats})
            log_metadata(metadata={
                f"iter_{len(metrics_history)-1}/mean_reward": float(stats["mean_reward"]),
                f"iter_{len(metrics_history)-1}/sps": float(stats["sps"]),
            })

    total_steps = trainer.global_step
    trainer.utilization.stop()
    vecenv.close()

    # ── Save checkpoint ───────────────────────────────────────────
    checkpoint_path = Path(tempfile.mkdtemp()) / f"{config.tag}_policy.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": config.model_dump(),
        "metrics_history": metrics_history,
    }, checkpoint_path)

    final = metrics_history[-1] if metrics_history else {}
    result = TrainingResult(
        env_name=config.env_name,
        tag=config.tag,
        mean_reward=best_reward,
        mean_episode_length=final.get("mean_episode_length", 0),
        total_timesteps=total_steps,
        steps_per_second=final.get("sps", 0),
        policy_loss=final.get("policy_loss", 0),
        value_loss=final.get("value_loss", 0),
        entropy=final.get("entropy", 0),
        checkpoint_path=str(checkpoint_path),
        config=config.model_dump(),
        metrics_history=metrics_history,
    )

    log_metadata(
        metadata={
            "env": config.env_name,
            "best_reward": float(best_reward),
            "total_steps": int(total_steps),
            "learning_rate": float(config.learning_rate),
        },
        infer_artifact=True,
    )

    print(f"✅ {config.tag} → best reward: {best_reward:.2f}")
    return result


@step
def evaluate_agents(
    results: list[TrainingResult],
    eval_episodes: int = 100,
) -> Annotated[list[EvalResult], "eval_results"]:
    """
    Evaluate all trained policies and rank them.

    Fan-in step: receives ALL training results, loads each checkpoint,
    runs evaluation episodes, and produces a ranked list.
    """
    print(f"📊 Evaluating {len(results)} trained agents...")

    eval_results = []
    for result in results:
        vecenv, _, _ = _make_vecenv(
            result.env_name,
            vec={"num_envs": 1, "backend": "Serial"},
            env={"num_envs": 32},
        )
        policy = _make_policy(vecenv, checkpoint_path=result.checkpoint_path)
        rewards = _run_eval_episodes(policy, vecenv, eval_episodes)
        vecenv.close()

        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0

        eval_results.append(EvalResult(
            env_name=result.env_name,
            tag=result.tag,
            eval_mean_reward=mean_r,
            eval_std_reward=std_r,
            eval_episodes=len(rewards),
            checkpoint_path=result.checkpoint_path,
        ))
        print(f"   {result.tag}: {mean_r:.2f} ± {std_r:.2f}")

    # Mark the best per environment
    for env_name in set(r.env_name for r in eval_results):
        best = max(
            (r for r in eval_results if r.env_name == env_name),
            key=lambda r: r.eval_mean_reward,
        )
        best.is_best = True

    log_metadata(
        metadata={
            "leaderboard": {
                r.tag: {"reward": float(r.eval_mean_reward), "std": float(r.eval_std_reward)}
                for r in sorted(eval_results, key=lambda r: -r.eval_mean_reward)
            }
        },
        infer_artifact=True,
    )
    return eval_results


@step
def create_sweep_report(
    training_results: list[TrainingResult],
    eval_results: list[EvalResult],
) -> Annotated[HTMLString, "sweep_report"]:
    """
    Create an interactive HTML report: leaderboard table + Plotly training curves.

    Returns HTMLString for display in the ZenML dashboard.
    """
    import plotly.graph_objects as go

    # ── Leaderboard table ─────────────────────────────────────────
    sorted_evals = sorted(eval_results, key=lambda r: -r.eval_mean_reward)
    rows = []
    for r in sorted_evals:
        best_badge = " 🏆" if r.is_best else ""
        rows.append(
            f"<tr><td>{r.tag}</td><td>{r.eval_mean_reward:.2f} ± {r.eval_std_reward:.2f}</td>"
            f"<td>{r.eval_episodes}</td><td>{r.env_name}{best_badge}</td></tr>"
        )
    table_rows = "\n".join(rows)

    # ── Interactive training curves (Plotly) ──────────────────────
    curves_html = ""
    if training_results:
        # Chart 1: Mean Reward
        fig_reward = go.Figure()
        for result in training_results:
            hist = result.metrics_history or []
            if not hist:
                continue
            iters = [h.get("iteration", i) for i, h in enumerate(hist)]
            rewards = [h.get("mean_reward", 0) for h in hist]
            fig_reward.add_trace(go.Scatter(
                x=iters, y=rewards, mode="lines", name=result.tag, opacity=0.8,
            ))
        fig_reward.update_layout(
            title="Training: Mean Reward",
            xaxis_title="Iteration", yaxis_title="Mean Reward",
            height=400, template="plotly_white",
        )

        # Chart 2: Steps per Second
        fig_sps = go.Figure()
        for result in training_results:
            hist = result.metrics_history or []
            if not hist:
                continue
            iters = [h.get("iteration", i) for i, h in enumerate(hist)]
            sps = [h.get("sps", 0) for h in hist]
            fig_sps.add_trace(go.Scatter(
                x=iters, y=sps, mode="lines", name=result.tag, opacity=0.8,
            ))
        fig_sps.update_layout(
            title="Training: Steps per Second",
            xaxis_title="Iteration", yaxis_title="Steps/sec",
            height=400, template="plotly_white",
        )

        # Chart 3: Loss Curves (policy_loss, value_loss, entropy)
        fig_loss = go.Figure()
        for result in training_results:
            hist = result.metrics_history or []
            if not hist:
                continue
            iters = [h.get("iteration", i) for i, h in enumerate(hist)]
            for metric, dash in [("policy_loss", None), ("value_loss", "dash"), ("entropy", "dot")]:
                vals = [h.get(metric, 0) for h in hist]
                fig_loss.add_trace(go.Scatter(
                    x=iters, y=vals, mode="lines",
                    name=f"{result.tag} — {metric}",
                    line=dict(dash=dash), opacity=0.8,
                ))
        fig_loss.update_layout(
            title="Training: Loss Curves",
            xaxis_title="Iteration", yaxis_title="Value",
            height=400, template="plotly_white",
        )

        reward_html = fig_reward.to_html(full_html=False, include_plotlyjs="cdn")
        sps_html = fig_sps.to_html(full_html=False, include_plotlyjs=False)
        loss_html = fig_loss.to_html(full_html=False, include_plotlyjs=False)
        curves_html = f"""
        <h3>Training Curves</h3>
        <div style="margin: 1rem 0;">{reward_html}</div>
        <div style="margin: 1rem 0;">{sps_html}</div>
        <div style="margin: 1rem 0;">{loss_html}</div>
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
        {curves_html}
    </div>
    """
    return HTMLString(html)


@step
def promote_best_policy(
    eval_results: list[EvalResult],
) -> Annotated[dict, "promoted_policies"]:
    """
    Promote the best policy via ZenML's Model Control Plane.

    This step demonstrates the full model lifecycle:
    1. The current model version (created by this pipeline run) holds all artifacts
    2. We set the model version stage to "production" — visible in the dashboard
    3. Metadata is attached so you can trace: promoted model → eval results →
       training run → dataset → client/project (the full lineage for compliance)

    In the dashboard, click any model version to see:
    - Stage transitions (staging → production)
    - All linked artifacts (datasets, checkpoints, eval results)
    - The exact pipeline run, configs, and data that produced it
    """
    winners = [r for r in eval_results if r.is_best]

    promoted = {}
    for winner in winners:
        print(f"🏆 Promoting {winner.tag} → "
              f"reward {winner.eval_mean_reward:.2f} ± {winner.eval_std_reward:.2f}")

        promoted[winner.env_name] = {
            "tag": winner.tag,
            "eval_mean_reward": winner.eval_mean_reward,
            "eval_std_reward": winner.eval_std_reward,
            "checkpoint": winner.checkpoint_path,
        }

    # Transition model version stage — this is what shows up in the
    # Model Control Plane dashboard as a stage badge (staging/production)
    client = Client()
    latest_version = client.get_model_version(
        model_name_or_id="rl_policy",
    )
    client.update_model_version(
        model_name_or_id="rl_policy",
        version_name_or_id=latest_version.name,
        stage=ModelStages.PRODUCTION,
        force=True,
    )
    print(f"📋 Model version '{latest_version.name}' → stage: production")

    # Attach promoted policies as metadata — visible on model detail page
    log_metadata(
        metadata={
            "promoted_policies": promoted,
            "stage_transition": "staging → production",
            "promotion_criteria": "highest eval reward per environment",
        },
        infer_model=True,
    )
    return promoted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE DYNAMIC PIPELINE — This is where the magic happens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

docker_settings = DockerSettings(
    dockerfile="Dockerfile",
    python_package_installer="pip",
    parent_image_build_config=DockerBuildConfig(
        build_options=DockerBuildOptions(platform="linux/amd64"),
    ),
)

kubernetes_settings = KubernetesOrchestratorSettings(
    orchestrator_pod_settings={
        "resources": {
            "requests": {
                "cpu": "1",
                "memory": "4Gi",
                "ephemeral-storage": "20Gi",
            },
            "limits": {
                "cpu": "2",
                "memory": "8Gi",
                "ephemeral-storage": "30Gi",
            },
        },
    },
    pod_settings={
        "resources": {
            "requests": {
                "cpu": "2",
                "memory": "8Gi",
                "ephemeral-storage": "20Gi",
            },
            "limits": {
                "cpu": "4",
                "memory": "16Gi",
                "ephemeral-storage": "30Gi",
            },
        },
    },
)

@pipeline(
    dynamic=True,
    enable_cache=False,
    model=Model(name="rl_policy", license="MIT", description="PufferLib RL agents across multiple environments"),
    settings={"docker": docker_settings, "orchestrator": kubernetes_settings},
)
def rl_environment_sweep(
    env_names: list[str],
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
    # Data lineage / governance params
    client_id: str = "acme-corp",
    project: str = "rl-optimization",
    data_source: str = "internal-simulation",
    domain: str = "operations-research",
):
    """
    Dynamic RL training pipeline with PufferLib.

    Key properties:
    1. DATA LINEAGE: Every run traces back to a versioned dataset artifact
       with client/project metadata — enabling GDPR-style data scrubbing.
       "Delete all data for Client X" → find every artifact and model version.
    2. DYNAMIC: The number of training steps is determined at runtime
       by configure_sweep — not hardcoded in the pipeline definition.
    3. FAN-OUT: train_agent.map() creates one isolated step per config,
       each with its own container, GPU, artifacts, and retry logic.
    4. FAN-IN: evaluate_agents receives ALL results and compares them.
    5. MODEL CONTROL PLANE: Policies are versioned and promoted through
       stages (staging → production) with full lineage in the dashboard.
    6. INFRA-AGNOSTIC: Same pipeline code runs on any infrastructure.
       Switch with `zenml stack set`:
         - Local dev:  zenml stack set local
         - Kubernetes: zenml stack set k8s-gpu
         - Slurm/HPC:  zenml stack set slurm-cluster
       The pipeline code never changes — only the stack definition.
    7. RBAC: ZenML supports role-based access control — researchers can
       share pipelines, models, and artifacts without stepping on each
       other. Critical when onboarding 25-75 team members.

    Note on PufferLib environments:
        This demo uses PufferLib's built-in Ocean envs (pong, breakout, etc.)
        for convenience. Your custom environments plug in the same way —
        just register them with PufferLib and pass the env name here.
    """
    # Step 1: Register training data with lineage metadata
    # This artifact is the root of the lineage graph — traceable for compliance
    dataset = load_training_data(
        env_names=env_names,
        client_id=client_id,
        project=project,
        data_source=data_source,
        domain=domain,
    )

    # Step 2: Generate sweep configs at runtime (reads env_names from dataset)
    configs = configure_sweep(
        dataset=dataset,
        learning_rates=learning_rates,
        total_timesteps=total_timesteps,
        device=device,
    )

    # Step 3: Fan out — one training step per config
    # Each becomes an independent, trackable step in the DAG
    training_results = train_agent.map(configs)

    # Step 4: Fan in — compare all trained policies
    eval_results = evaluate_agents(training_results)

    # Step 5: Create HTML report (leaderboard + training curves)
    create_sweep_report(training_results=training_results, eval_results=eval_results)

    # Step 6: Promote the winner(s) — stage transition visible in Model Control Plane
    promote_best_policy(eval_results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN IT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Device: detect locally but note that remote K8s pods will use whatever
    # value is passed here. Use "cpu" unless your cluster has GPU nodes.
    device = "cpu"

    # ── Example: sweep across 3 environments × 2 learning rates ───
    # This creates 6 parallel training runs dynamically.
    # Note: We use PufferLib's built-in Ocean envs here, but your custom
    # environments plug in the same way — just register with PufferLib.
    rl_environment_sweep(
        env_names=[
            "ocean-pong",         # Fast Atari-like env from PufferLib Ocean
            "ocean-breakout",     # Another classic, C-based for speed
            "ocean-connect4",     # Board game — different reward structure
        ],
        learning_rates=[3e-4, 1e-3],
        total_timesteps=100_000,
        device=device,
        # Data lineage: these tags make every artifact traceable to a client/project
        client_id="acme-corp",
        project="rl-optimization",
        data_source="internal-simulation",
        domain="operations-research",
    )

    # ── Or: single environment, hyperparam sweep ──────────────────
    # rl_environment_sweep(
    #     env_names=["ocean-snake"],
    #     learning_rates=[1e-4, 3e-4, 1e-3, 3e-3],
    #     total_timesteps=2_000_000,
    #     client_id="client-beta",
    #     project="logistics-routing",
    # )

    # ── Or: pull config from a YAML / database / API ──────────────
    # The pipeline shape adapts — that's the whole point of dynamic.

    # ── Infra-agnostic: same code, different stack ─────────────────
    # zenml stack set local          → runs on your laptop
    # zenml stack set k8s-gpu        → fans out to Kubernetes pods with GPUs
    # zenml stack set slurm-cluster  → submits to HPC job scheduler
    # The pipeline code above doesn't change. Only the stack definition.