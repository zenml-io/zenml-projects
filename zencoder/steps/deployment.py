from typing import Dict, Optional, Tuple, cast

from zenml import get_step_context, step
from zenml.client import Client
from zenml.integrations.huggingface.model_deployers import (
    HuggingFaceModelDeployer,
)
from zenml.integrations.huggingface.services import (
    HuggingFaceDeploymentService,
    HuggingFaceServiceConfig,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


def parse_huggingface_url(url: str) -> Tuple[str, str, str]:
    """
    Parses a Hugging Face Hub URL to extract the namespace, repository, and revision.

    Args:
        url: The Hugging Face Hub URL to parse. Expected format:
             "https://huggingface.co/{namespace}/{repository}/tree/{revision}".

    Returns:
        A tuple containing:
        - namespace: The owner or organization of the repository.
        - repository: The name of the repository.
        - revision: The specific commit hash or branch name.

    Raises:
        ValueError: If the URL does not match the expected format.
    """
    # Split the URL into parts
    parts = url.split("/")

    # Check if the URL has the expected number of parts
    if len(parts) >= 7 and parts[2] == "huggingface.co":
        # Extract the namespace (owner), repository, and revision (commit hash)
        namespace = parts[3]
        repository = parts[4]
        revision = parts[6]
        return namespace, repository, revision
    else:
        # Raise an error if the URL doesn't have the expected format
        raise ValueError("Invalid Huggingface URL format")


@step(enable_cache=False)
def deploy_model_to_hf_hub(hf_endpoint_cfg: Optional[Dict] = None) -> None:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        hf_endpoint_cfg: The configuration for the Huggingface endpoint.

    """
    # Endpoint name is managed by the HuggingFace service
    hf_endpoint_cfg = HuggingFaceServiceConfig(**hf_endpoint_cfg)

    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]

    commit_info = get_step_context().model.run_metadata[
        "merged_model_commit_info"
    ]

    model_namespace, repository, revision = parse_huggingface_url(commit_info)

    if repository is None:
        raise ValueError(
            "The ZenML model version does not have a repository in its metadata. "
            "Please make sure that the training pipeline is configured correctly."
        )

    if (
        hf_endpoint_cfg.repository is None
        or hf_endpoint_cfg.revision is None
        or hf_endpoint_cfg.token is None
    ):
        logger.warning(
            "The Huggingface endpoint configuration has already been set via an old pipeline run. "
            "The endpoint name, repository, and revision will be overwritten."
        )
        hf_endpoint_cfg.repository = f"{model_namespace}/{repository}"
        hf_endpoint_cfg.revision = revision
        hf_endpoint_cfg.token = hf_token

    # TODO: Can check if the model deployer is of the right type
    model_deployer = cast(
        HuggingFaceModelDeployer,
        HuggingFaceModelDeployer.get_active_model_deployer(),
    )
    service = cast(
        HuggingFaceDeploymentService,
        model_deployer.deploy_model(config=hf_endpoint_cfg),
    )

    logger.info(
        f"Huggingface Inference Endpoint deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )
