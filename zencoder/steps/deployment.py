from zenml import step
from zenml.client import Client
from zenml import ArtifactConfig
from typing_extensions import Annotated
from zenml import get_step_context
from typing import Optional, Dict, cast
import random
from zenml import log_artifact_metadata
from zenml.logger import get_logger
from huggingface.hf_deployment import (
    HuggingFaceModelService,
    HFInferenceEndpointConfig,
)
from huggingface.hf_model_deployer import HFEndpointModelDeployer

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
    framework: str,
    accelerator: str,
    instance_size: str,
    instance_type: str,
    region: str,
    vendor: str,
    endpoint_name: Optional[str] = None,
    account_id: Optional[str] = None,
    min_replica: int = 0,
    max_replica: int = 1,
    task: Optional[str] = None,
    namespace: Optional[str] = None,
    custom_image: Optional[Dict] = None,
    endpoint_type: str = "public",
) -> Annotated[
    HuggingFaceModelService,
    ArtifactConfig(name="endpoint", is_deployment_artifact=True),
]:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        framework (str): The framework of the model.
        accelerator (str): The accelerator of the model.
        instance_size (str): The instance size of the model.
        instance_type (str): The instance type of the model.
        region (str): The region of the model.
        vendor (str): The vendor of the model.
        endpoint_name (Optional[str]): The name of the model.
        account_id (Optional[str]): The account id of the model.
        min_replica (int): The minimum replica of the model.
        max_replica (int): The maximum replica of the model.
        task (Optional[str]): The task of the model.
        namespace (Optional[str]): Namespace of the organization.
        custom_image (Optional[Dict]): The custom image of the model.
        endpoint_type (str): The type of the model.

    """
    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]
    # commit_info = get_step_context().model_version.metadata[
    #     "merged_model_commit_info"
    # ]
    commit_info = "https://huggingface.co/htahir1/peft-lora-zencoder15B-personal-copilot-merged/commit/e661d8219e050a23eba54cb44f8e93d5f11885d2"
    model_namespace, repository, revision = parse_huggingface_url(commit_info)

    if repository is None:
        raise ValueError(
            "The ZenML model version does not have a repository in its metadata. "
            "Please make sure that the training pipeline is configured correctly."
        )

    if endpoint_name is None:
        endpoint_name = generate_random_letters()

    hf_endpoint_cfg = HFInferenceEndpointConfig(
        endpoint_name=endpoint_name,
        repository=f"{model_namespace}/{repository}",
        framework=framework,
        accelerator=accelerator,
        instance_size=instance_size,
        instance_type=instance_type,
        region=region,
        vendor=vendor,
        hf_token=hf_token,
        account_id=account_id,
        min_replica=min_replica,
        max_replica=max_replica,
        revision=revision,
        task=task,
        custom_image=custom_image,
        namespace=namespace,
        endpoint_type=endpoint_type,
    )

    model_deployer = cast(
        HFEndpointModelDeployer,
        HFEndpointModelDeployer.get_active_model_deployer(),
    )
    service = cast(
        HuggingFaceModelService,
        model_deployer.deploy_model(config=hf_endpoint_cfg),
    )

    log_artifact_metadata(metadata={"deployment_service": service.to_dict()})

    logger.info(
        f"Huggingface Inference Endpoint deployment service started and reachable at:\n"
        f"    {service.prediction_url}\n"
    )

    return service
