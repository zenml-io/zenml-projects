"""Evaluation pipeline for the email search agent.

This pipeline evaluates a trained model on test scenarios:
1. Loads the trained model from checkpoint
2. Runs inference on each test scenario
3. Computes accuracy and other metrics

Requires GPU resources for inference.
"""

from typing import List

from environment.models import Scenario
from steps.evaluation import (
    compute_metrics,
    load_trained_model,
    run_inference,
)
from zenml import Model, pipeline
from zenml.config import DockerSettings

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
def evaluation_pipeline(
    model_config: dict,
    checkpoint_path: str,
    test_scenarios: List[Scenario],
    db_path: str,
    judge_model: str = "openai/gpt-4.1",
    art_path: str = "./.art",
):
    """Evaluate the trained email search agent.

    Args:
        model_config: Model configuration from training.
        checkpoint_path: Path to the trained checkpoint.
        test_scenarios: Test scenarios from data preparation.
        db_path: Path to the email database.
        judge_model: LiteLLM model for correctness judging.
        art_path: Directory for ART files.
    """
    # Step 1: Prepare model loading configuration
    inference_config = load_trained_model(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        art_path=art_path,
    )

    # Step 2: Run inference on test scenarios (requires GPU)
    predictions = run_inference(
        inference_config=inference_config,
        test_scenarios=test_scenarios,
        db_path=db_path,
        judge_model=judge_model,
    )

    # Step 3: Compute metrics
    metrics = compute_metrics(predictions=predictions)

    return predictions, metrics
