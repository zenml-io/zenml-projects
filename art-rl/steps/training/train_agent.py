"""Step to run the ART training loop with GRPO and RULER scoring."""

import asyncio
from typing import Annotated, List, Tuple

from environment.models import Scenario
from zenml import log_model_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


def _run_training_loop(
    model_config: dict,
    train_scenarios: List[Scenario],
    db_path: str,
    groups_per_step: int,
    num_epochs: int,
    rollouts_per_group: int,
    learning_rate: float,
    max_steps: int,
    ruler_model: str,
    art_path: str,
) -> Tuple[str, dict]:
    """Internal async training loop wrapped for sync execution.

    This function contains the core ART training logic using GRPO
    (Group Relative Policy Optimization) with RULER scoring.
    """
    import art
    from agent.rollout import rollout
    from art.langgraph import wrap_rollout
    from art.local import LocalBackend
    from art.rewards import ruler_score_group
    from art.utils import iterate_dataset
    from environment.models import EmailScenario

    async def _async_train():
        # Initialize model with memory-optimized config for GPU containers
        model = art.TrainableModel(
            name=model_config["name"],
            project=model_config["project"],
            base_model=model_config["base_model"],
        )

        # Configure for GPU memory efficiency
        model._internal_config = art.dev.InternalModelConfig(
            init_args=art.dev.InitArgs(max_seq_length=8192),
            engine_args=art.dev.EngineArgs(
                enforce_eager=True,
                gpu_memory_utilization=0.8,
            ),
        )

        # LocalBackend runs inference/training in the same process
        backend = LocalBackend(in_process=True, path=art_path)
        await model.register(backend)

        # Training iterator handles epoch cycling and batching
        training_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=groups_per_step,
            num_epochs=num_epochs,
            initial_step=await model.get_step(),
        )

        all_metrics = []

        for batch in training_iterator:
            logger.info(
                f"Training step {batch.step}, epoch {batch.epoch}, "
                f"batch size {len(batch.items)}"
            )

            # Create trajectory groups - each scenario gets multiple rollouts
            groups = []
            for scenario in batch.items:
                email_scenario = EmailScenario(
                    step=batch.step,
                    scenario=scenario,
                )
                groups.append(
                    art.TrajectoryGroup(
                        (
                            wrap_rollout(model, rollout)(
                                model, email_scenario, db_path=db_path
                            )
                            for _ in range(rollouts_per_group)
                        )
                    )
                )

            # Gather all trajectories (parallel execution)
            finished_groups = await art.gather_trajectory_groups(
                groups,
                pbar_desc="rollouts",
                max_exceptions=rollouts_per_group * len(batch.items),
            )

            # RULER scoring - uses LLM judge to rank trajectories
            judged_groups = []
            for group in finished_groups:
                judged_group = await ruler_score_group(group, ruler_model)
                if judged_group:
                    judged_groups.append(judged_group)

            if not judged_groups:
                logger.warning(
                    "No valid trajectory groups after RULER scoring"
                )
                continue

            # Calculate metrics for logging
            total_trajectories = sum(
                len(g.trajectories) for g in judged_groups
            )
            avg_reward = (
                sum(t.reward for g in judged_groups for t in g.trajectories)
                / total_trajectories
            )
            accuracy = (
                sum(
                    t.metrics.get("correct", 0)
                    for g in judged_groups
                    for t in g.trajectories
                )
                / total_trajectories
            )

            step_metrics = {
                "step": batch.step,
                "epoch": batch.epoch,
                "avg_reward": avg_reward,
                "accuracy": accuracy,
                "num_trajectories": total_trajectories,
            }
            all_metrics.append(step_metrics)

            # Log to ZenML Model Control Plane
            log_model_metadata(metadata=step_metrics)
            logger.info(
                f"Step {batch.step}: reward={avg_reward:.3f}, "
                f"accuracy={accuracy:.3f}"
            )

            # GRPO training step
            await model.delete_checkpoints()
            await model.train(
                judged_groups,
                config=art.TrainConfig(learning_rate=learning_rate),
                _config={"logprob_calculation_chunk_size": 8},
            )

            if batch.step >= max_steps:
                logger.info(f"Reached max_steps={max_steps}, stopping")
                break

        checkpoint_path = f"{art_path}/checkpoints/latest"
        return checkpoint_path, {"history": all_metrics}

    return asyncio.run(_async_train())


@step(enable_cache=False)
def train_agent(
    model_config: dict,
    train_scenarios: List[Scenario],
    db_path: str,
    groups_per_step: int = 2,
    num_epochs: int = 20,
    rollouts_per_group: int = 4,
    learning_rate: float = 1e-5,
    max_steps: int = 20,
    ruler_model: str = "openai/o4-mini",
    art_path: str = "./.art",
) -> Tuple[
    Annotated[str, "checkpoint_path"],
    Annotated[dict, "training_metrics"],
]:
    """Run the ART training loop for the email search agent.

    This step implements the core RL training using:
    - GRPO (Group Relative Policy Optimization) for policy updates
    - RULER for scoring trajectories relative to each other
    - LangGraph ReAct agents for executing rollouts

    The training process:
    1. For each batch of scenarios, generate multiple rollout trajectories
    2. Use RULER to score trajectories relative to each other
    3. Apply GRPO update using the scored trajectories
    4. Log metrics to ZenML for tracking

    Args:
        model_config: Configuration from setup_art_model step.
        train_scenarios: List of training scenarios.
        db_path: Path to the email database.
        groups_per_step: Number of scenario groups per training step.
        num_epochs: Total number of passes through the training data.
        rollouts_per_group: Number of rollouts per scenario for GRPO.
        learning_rate: Learning rate for the optimizer.
        max_steps: Maximum training steps (for early stopping).
        ruler_model: LiteLLM model ID for RULER scoring.
        art_path: Directory for ART checkpoints and logs.

    Returns:
        Tuple of (checkpoint_path, training_metrics).
    """
    logger.info("Starting ART training loop...")
    logger.info(f"Training on {len(train_scenarios)} scenarios")
    logger.info(
        f"Config: groups_per_step={groups_per_step}, "
        f"rollouts_per_group={rollouts_per_group}"
    )

    checkpoint_path, metrics = _run_training_loop(
        model_config=model_config,
        train_scenarios=train_scenarios,
        db_path=db_path,
        groups_per_step=groups_per_step,
        num_epochs=num_epochs,
        rollouts_per_group=rollouts_per_group,
        learning_rate=learning_rate,
        max_steps=max_steps,
        ruler_model=ruler_model,
        art_path=art_path,
    )

    logger.info(f"Training complete. Checkpoint saved to {checkpoint_path}")
    return checkpoint_path, metrics
