"""
🐡 PufferLib x ZenML: Dynamic RL Training Pipeline
====================================================

A demo showing how ZenML's dynamic pipelines orchestrate PufferLib RL
training across multiple environments, with automatic experiment tracking,
artifact versioning, and policy promotion — all in one pipeline.

This is what "MLOps for RL" looks like when you don't want to pay £100K
for experiment tracking alone.

Requirements:
    pip install "zenml[server]" pufferlib torch

Architecture:
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
    │  promote_best    │  ← Version & promote the winning policy
    └─────────────────┘
"""

import base64
import io
import json
import tempfile
from pathlib import Path

from pydantic import BaseModel
import torch
import torch.nn as nn

# ─── ZenML imports ───────────────────────────────────────────────────
from zenml import Model, pipeline, step, log_metadata
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.types import HTMLString

# ─── PufferLib imports ───────────────────────────────────────────────
import pufferlib
from pufferlib.pufferl import PuffeRL
import pufferlib.pufferl as pufferl
from pufferlib.pytorch import layer_init


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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
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
# HELPERS — PufferLib 3.0 env naming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _puffer_env_name(name: str) -> str:
    """Map 'ocean-pong' → 'puffer_pong' for PufferLib config lookup."""
    if name.startswith("ocean-"):
        return "puffer_" + name[6:]
    return name


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEPS — Each is independently tracked, cached, and retriable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@step
def configure_sweep(
    env_names: list[str],
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
) -> list[EnvConfig]:
    """
    Generate the sweep configuration at runtime.

    This is the "dynamic" part — the number of downstream training steps
    is determined HERE, not hardcoded in the pipeline definition.
    You could also pull this from a config file, database, or API.
    """
    configs = []
    for env_name in env_names:
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
            "environments": env_names,
            "learning_rates": learning_rates,
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
) -> TrainingResult:
    """
    Train a single RL agent using PufferLib's PuffeRL trainer.

    This step runs independently per environment config. ZenML handles:
    - Containerization (same Docker image, different params)
    - GPU allocation via the orchestrator (K8s, Slurm, local)
    - Artifact versioning of the trained checkpoint
    - Automatic retry on failure
    """
    print(f"🎮 Training on {config.env_name} | lr={config.learning_rate}")

    # ── Resolve PufferLib env name ─────────────────────────────────
    puffer_name = _puffer_env_name(config.env_name)

    # ── Load config and create vectorized env ──────────────────────
    args = pufferl.load_config(puffer_name)
    args["vec"]["num_envs"] = max(1, config.num_workers)
    args["vec"]["backend"] = "Serial" if config.num_workers <= 1 else "Multiprocessing"
    if args["vec"]["backend"] == "Multiprocessing":
        args["vec"]["num_workers"] = config.num_workers
    args["env"]["num_envs"] = config.num_envs
    args["train"]["device"] = config.device
    args["train"]["learning_rate"] = config.learning_rate
    args["train"]["batch_size"] = config.batch_size
    args["train"]["total_timesteps"] = config.total_timesteps
    args["train"]["update_epochs"] = config.update_epochs
    args["train"]["optimizer"] = "adam"  # avoid heavyball dependency
    args["train"]["use_rnn"] = False  # our RLPolicy is MLP-only, no LSTM
    args["train"]["minibatch_size"] = min(4096, config.batch_size)
    args["train"]["max_minibatch_size"] = config.batch_size

    vecenv = pufferl.load_env(puffer_name, args)

    # ── Build policy ──────────────────────────────────────────────
    obs_shape = vecenv.single_observation_space.shape[0]
    act_shape = vecenv.single_action_space.n
    policy = RLPolicy(obs_shape, act_shape).to(config.device)

    # ── Configure PuffeRL trainer (PufferLib 3.0 API) ──────────────
    train_config = dict(**args["train"], env=puffer_name)
    trainer = PuffeRL(train_config, vecenv, policy)

    # ── Training loop ─────────────────────────────────────────────
    total_steps = 0
    best_reward = float("-inf")
    metrics_history = []

    while trainer.global_step < config.total_timesteps:
        trainer.evaluate()
        logs = trainer.train()
        total_steps = trainer.global_step

        # PuffeRL logs use keys like 'environment/mean_reward', 'losses/policy_loss', 'SPS'
        def _get(key: str, alt: str = None):
            if logs is None:
                return 0
            return logs.get(key, logs.get(alt or key, 0))

        stats = {
            "mean_reward": _get("environment/mean_reward", "environment/score"),
            "mean_episode_length": _get("environment/mean_episode_length", "environment/episode_length"),
            "policy_loss": _get("losses/policy_loss"),
            "value_loss": _get("losses/value_loss"),
            "entropy": _get("losses/entropy"),
            "sps": _get("SPS"),
        }
        if stats["mean_reward"] > best_reward:
            best_reward = stats["mean_reward"]

        if logs is not None:
            metrics_history.append({
                "iteration": len(metrics_history),
                **stats,
            })
            log_metadata(metadata={
                f"iter_{len(metrics_history)-1}/mean_reward": stats["mean_reward"],
                f"iter_{len(metrics_history)-1}/sps": stats["sps"],
            })

    # ── Save checkpoint ───────────────────────────────────────────
    checkpoint_dir = Path(tempfile.mkdtemp())
    checkpoint_path = checkpoint_dir / f"{config.tag}_policy.pt"
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": config.model_dump(),
        "metrics_history": metrics_history,
    }, checkpoint_path)

    # ── Compose result ────────────────────────────────────────────
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

    # Log final summary metadata — infer_artifact attaches to this step's output
    log_metadata(
        metadata={
            "env": config.env_name,
            "best_reward": best_reward,
            "total_steps": total_steps,
            "learning_rate": config.learning_rate,
        },
        infer_artifact=True,
    )

    print(f"✅ {config.tag} → best reward: {best_reward:.2f}")
    return result


@step
def evaluate_agents(
    results: list[TrainingResult],
    eval_episodes: int = 100,
) -> list[EvalResult]:
    """
    Evaluate all trained policies and rank them.

    Fan-in step: receives ALL training results, loads each checkpoint,
    runs evaluation episodes, and produces a ranked list.
    """
    print(f"📊 Evaluating {len(results)} trained agents...")

    eval_results = []
    for result in results:
        # Load checkpoint
        checkpoint = torch.load(result.checkpoint_path, weights_only=False)

        # Recreate vecenv for evaluation (Serial backend, small batch)
        puffer_name = _puffer_env_name(result.env_name)
        args = pufferl.load_config(puffer_name)
        args["vec"]["num_envs"] = 1
        args["vec"]["backend"] = "Serial"
        args["env"]["num_envs"] = 32
        vecenv = pufferl.load_env(puffer_name, args)

        obs_shape = vecenv.single_observation_space.shape[0]
        act_shape = vecenv.single_action_space.n

        policy = RLPolicy(obs_shape, act_shape)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()

        # Run evaluation episodes using PufferLib vector API
        rewards = []
        obs, _ = vecenv.reset()
        episode_rewards = [0.0] * vecenv.num_agents
        completed = 0

        while completed < eval_episodes:
            with torch.no_grad():
                logits, _ = policy.forward_eval(torch.FloatTensor(obs))
                actions = logits.argmax(dim=-1).numpy()
            obs, reward, terminated, truncated, info = vecenv.step(actions)
            n = vecenv.num_agents
            for i in range(n):
                episode_rewards[i] += float(reward[i])
                if terminated[i] or truncated[i]:
                    rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
                    completed += 1

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
    envs = set(r.env_name for r in eval_results)
    for env_name in envs:
        env_results = [r for r in eval_results if r.env_name == env_name]
        best = max(env_results, key=lambda r: r.eval_mean_reward)
        best.is_best = True

    # Log leaderboard metadata — infer_artifact attaches to eval_results output
    log_metadata(
        metadata={
            "leaderboard": {
                r.tag: {"reward": r.eval_mean_reward, "std": r.eval_std_reward}
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
) -> HTMLString:
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

    # ── Training curves (matplotlib → base64) ────────────────────────
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
        curve_html = f'''
        <h3>Training Curves</h3>
        <div style="margin: 1rem 0;">
            <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; height: auto;">
        </div>
        '''

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


@step
def promote_best_policy(
    eval_results: list[EvalResult],
) -> dict:
    """
    Promote the best policy as a versioned ZenML Model artifact.

    Uses ZenML Model Control Plane:
    - Policies are linked to the model via model context
    - Metadata is attached to the model version (infer_model=True)
    - Full lineage and reproducibility in the dashboard
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

    # Attach promoted policies to the ZenML Model — visible on model detail page
    log_metadata(
        metadata={"promoted_policies": promoted},
        infer_model=True,
    )
    return promoted


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE DYNAMIC PIPELINE — This is where the magic happens
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pipeline(
    dynamic=True,
    enable_cache=False,
    model=Model(name="rl_policy", license="MIT", description="PufferLib RL agents across multiple environments"),
)
def rl_environment_sweep(
    env_names: list[str],
    learning_rates: list[float],
    total_timesteps: int = 100_000,
    device: str = "cuda",
):
    """
    Dynamic RL training pipeline with PufferLib.

    Key properties:
    1. DYNAMIC: The number of training steps is determined at runtime
       by configure_sweep — not hardcoded in the pipeline definition.
    2. FAN-OUT: train_agent.map() creates one isolated step per config,
       each with its own container, GPU, artifacts, and retry logic.
    3. FAN-IN: evaluate_agents receives ALL results and compares them.
    4. MODEL: All artifacts are linked to the ZenML Model "rl_policy".
    5. INFRA-AGNOSTIC: Same code runs locally, on K8s, or on Slurm.
       Just switch the ZenML stack.
    """
    # Step 1: Generate sweep configs at runtime
    configs = configure_sweep(
        env_names=env_names,
        learning_rates=learning_rates,
        total_timesteps=total_timesteps,
        device=device,
    )

    # Step 2: Fan out — one training step per config
    # Each becomes an independent, trackable step in the DAG
    training_results = train_agent.map(configs)

    # Step 3: Fan in — compare all trained policies
    eval_results = evaluate_agents(training_results)

    # Step 4: Create HTML report (leaderboard + training curves)
    create_sweep_report(training_results=training_results, eval_results=eval_results)

    # Step 5: Promote the winner(s) — metadata attached to Model via infer_model
    promote_best_policy(eval_results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN IT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Device: cuda on GPU, mps on Apple Silicon, else cpu
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # ── Example: sweep across 3 environments × 2 learning rates ───
    # This creates 6 parallel training runs dynamically
    rl_environment_sweep(
        env_names=[
            "ocean-pong",         # Fast Atari-like env from PufferLib Ocean
            "ocean-breakout",     # Another classic, C-based for speed
            "ocean-connect4",     # Board game — different reward structure
        ],
        learning_rates=[3e-4, 1e-3],
        total_timesteps=100_000,
        device=device,
    )

    # ── Or: single environment, hyperparam sweep ──────────────────
    # rl_environment_sweep(
    #     env_names=["ocean-snake"],
    #     learning_rates=[1e-4, 3e-4, 1e-3, 3e-3],
    #     total_timesteps=2_000_000,
    # )

    # ── Or: pull config from a YAML / database / API ──────────────
    # The pipeline shape adapts — that's the whole point of dynamic.