"""Train a single RL agent using PufferLib's PuffeRL trainer."""

from typing import Annotated, Tuple

from materializers.policy_checkpoint_materializer import (
    PolicyCheckpointMaterializer,
)
from pufferlib.pufferl import PuffeRL
from steps.helpers import (
    extract_logs,
    make_policy,
    make_vecenv,
    resolve_device,
)
from steps.models import EnvConfig, PolicyCheckpoint, TrainingResult
from zenml import ArtifactConfig, log_metadata, step
from zenml.enums import ArtifactType
from zenml.materializers import PydanticMaterializer
from zenml.types import HTMLString


@step(
    output_materializers={
        "training_result": PydanticMaterializer,
        "policy_checkpoint": PolicyCheckpointMaterializer,
    },
    enable_cache=False,
)
def train_agent(
    config: EnvConfig,
) -> Tuple[
    Annotated[TrainingResult, "training_result"],
    Annotated[
        PolicyCheckpoint,
        ArtifactConfig(
            name="policy_checkpoint", artifact_type=ArtifactType.DATA
        ),
    ],
    Annotated[HTMLString, "training_summary"],
]:
    """
    Train a single RL agent using PufferLib's PuffeRL trainer.

    This step runs independently per environment config. ZenML handles:
    - Containerization (same Docker image, different params)
    - GPU allocation via the orchestrator (K8s, Slurm, local)
    - Artifact versioning of the trained checkpoint
    - Automatic retry on failure
    """
    print(f"🎮 Training on {config.env_name} | lr={config.learning_rate}")

    device = resolve_device(config.device)

    backend = "Serial" if config.num_workers <= 1 else "Multiprocessing"
    vec_overrides = {
        "num_envs": max(1, config.num_workers),
        "backend": backend,
    }
    if backend == "Multiprocessing":
        vec_overrides["num_workers"] = config.num_workers

    vecenv, puffer_name, args = make_vecenv(
        config.env_name,
        vec=vec_overrides,
        env={"num_envs": config.num_envs},
        train={
            "device": device,
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
    policy = make_policy(vecenv, device=device)
    trainer = PuffeRL(dict(**args["train"], env=puffer_name), vecenv, policy)

    best_reward = float("-inf")
    metrics_history = []

    while trainer.global_step < config.total_timesteps:
        trainer.evaluate()
        logs = trainer.train()

        stats = extract_logs(logs)
        if stats["mean_reward"] > best_reward:
            best_reward = stats["mean_reward"]

        if logs is not None:
            metrics_history.append(
                {"iteration": len(metrics_history), **stats}
            )
            log_metadata(
                metadata={
                    f"iter_{len(metrics_history) - 1}/mean_reward": float(
                        stats["mean_reward"]
                    ),
                    f"iter_{len(metrics_history) - 1}/sps": float(
                        stats["sps"]
                    ),
                }
            )

    total_steps = trainer.global_step
    trainer.utilization.stop()
    vecenv.close()

    checkpoint: PolicyCheckpoint = {
        "model_state_dict": policy.state_dict(),
        "config": config.model_dump(),
        "metrics_history": metrics_history,
    }

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
        artifact_name="training_result",
        infer_artifact=True,
    )

    print(f"✅ {config.tag} → best reward: {best_reward:.2f}")

    # HTML summary for ZenML dashboard
    final_sps = final.get("sps", 0)
    summary_html = HTMLString(f"""
    <div style="font-family: system-ui, sans-serif; padding: 1rem; border: 1px solid #eee; border-radius: 8px;">
        <h4>{config.tag}</h4>
        <table style="border-collapse: collapse; font-size: 0.9rem;">
            <tr><td style="padding: 0.25rem 0.5rem 0.25rem 0;"><b>Best reward</b></td><td>{best_reward:.2f}</td></tr>
            <tr><td style="padding: 0.25rem 0.5rem 0.25rem 0;"><b>Total steps</b></td><td>{total_steps:,}</td></tr>
            <tr><td style="padding: 0.25rem 0.5rem 0.25rem 0;"><b>Learning rate</b></td><td>{config.learning_rate}</td></tr>
            <tr><td style="padding: 0.25rem 0.5rem 0.25rem 0;"><b>Steps/sec</b></td><td>{final_sps:.1f}</td></tr>
        </table>
    </div>
    """)
    return result, checkpoint, summary_html
