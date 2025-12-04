"""Training pipeline for the email search agent.

This pipeline runs the ART training loop using:
- GRPO (Group Relative Policy Optimization) for policy updates
- RULER for relative trajectory scoring
- LangGraph ReAct agents for executing rollouts

Requires GPU resources for the train_agent step.
"""

from typing import List

from environment.models import Scenario
from steps.training import setup_art_model, train_agent
from zenml import Model, pipeline
from zenml.config import DockerSettings

# Docker settings for GPU training
docker_settings = DockerSettings(
    parent_image="pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime",
    requirements="requirements.txt",
    python_package_installer="uv",
    apt_packages=["git"],
)


@pipeline(
    model=Model(
        name="art-email-agent",
        description="Email search agent trained with ART + LangGraph",
        tags=["art", "langgraph", "email-agent", "rl"],
    ),
    settings={"docker": docker_settings},
)
def training_pipeline(
    train_scenarios: List[Scenario],
    db_path: str,
    model_name: str = "art-email-agent",
    project_name: str = "email-search-agent",
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    groups_per_step: int = 2,
    num_epochs: int = 20,
    rollouts_per_group: int = 4,
    learning_rate: float = 1e-5,
    max_steps: int = 20,
    ruler_model: str = "openai/o4-mini",
    art_path: str = "./.art",
):
    """Train the email search agent using ART.

    This pipeline:
    1. Configures the ART model
    2. Runs the training loop with GRPO and RULER

    Args:
        train_scenarios: Training scenarios from data preparation.
        db_path: Path to the email database.
        model_name: Name for the trained model.
        project_name: Project name for experiment tracking.
        base_model: Hugging Face model ID for the base model.
        groups_per_step: Scenario groups per training step.
        num_epochs: Number of passes through training data.
        rollouts_per_group: Rollouts per scenario for GRPO.
        learning_rate: Optimizer learning rate.
        max_steps: Maximum training steps.
        ruler_model: LiteLLM model for RULER scoring.
        art_path: Directory for ART checkpoints.
    """
    # Step 1: Configure the model
    model_config = setup_art_model(
        model_name=model_name,
        project_name=project_name,
        base_model=base_model,
    )

    # Step 2: Run training (requires GPU)
    checkpoint_path, training_metrics = train_agent(
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

    return checkpoint_path, training_metrics
