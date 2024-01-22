from zenml import step
from zenml import ArtifactConfig
from typing_extensions import Annotated
from zenml import get_step_context
from zenml.client import Client
from typing import Optional, cast, Dict
import random
from zenml import log_artifact_metadata
from zenml.logger import get_logger
from huggingface.hf_deployment_service import (
    HuggingFaceDeploymentService,
    HuggingFaceServiceConfig,
)
from huggingface.hf_model_deployer import HuggingFaceModelDeployer

logger = get_logger(__name__)


def generate_random_letters(number_of_letters: int = 10) -> str:
    """Generates three random letters.

    Returns:
        Three random letters.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(letters) for _ in range(number_of_letters))


def parse_huggingface_url(url):
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


@step
def deploy_model_to_hf_hub(
    hf_endpoint_cfg: Optional[Dict] = None,
) -> Annotated[
    HuggingFaceDeploymentService,
    ArtifactConfig(name="endpoint", is_deployment_artifact=True),
]:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        hf_endpoint_cfg: The configuration for the Huggingface endpoint.

    """
    endpoint_name = None
    hf_endpoint_cfg = HuggingFaceServiceConfig(**hf_endpoint_cfg)

    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]

    commit_info = get_step_context().model_version.metadata[
        "merged_model_commit_info"
    ]

    model_namespace, repository, revision = parse_huggingface_url(commit_info)

    if repository is None:
        raise ValueError(
            "The ZenML model version does not have a repository in its metadata. "
            "Please make sure that the training pipeline is configured correctly."
        )

    if endpoint_name is None:
        endpoint_name = generate_random_letters()

    if (
        hf_endpoint_cfg.endpoint_name is None
        or hf_endpoint_cfg.repository is None
        or hf_endpoint_cfg.revision is None
        or hf_endpoint_cfg.token is None
    ):
        logger.warning(
            "The Huggingface endpoint configuration has already been set via an old pipeline run. "
            "The endpoint name, repository, and revision will be overwritten."
        )
        hf_endpoint_cfg.endpoint_name = endpoint_name
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

    service_metadata = service.dict()
    # UUID object is not json serializable
    service_metadata["uuid"] = str(service_metadata["uuid"])
    log_artifact_metadata(metadata={"deployment_service": service_metadata})

    logger.info(
        f"Huggingface Inference Endpoint deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )

    return service
