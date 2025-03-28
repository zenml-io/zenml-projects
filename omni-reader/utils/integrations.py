"""This module contains the integrations for the project."""

from dotenv import load_dotenv
from huggingface_hub import login
from zenml.config import DockerSettings
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)

load_dotenv()

DOCKER_SETTINGS = DockerSettings(
    required_integrations=["opencv"],
    requirements=["opencv-python-headless"],
)

MLFLOW_SETTINGS = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"project": "ZenOCR", "models": "Gemma3-MistralAI-Comparison"},
)
