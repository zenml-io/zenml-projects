from zenml import step
from zenml.client import Client
from typing import cast, Annotated
from zenml import get_step_context
from huggingface.hf_model_deployer import HFEndpointModelDeployer
from huggingface.hf_deployment import (
    HuggingFaceModelService,
    HFInferenceEndpointConfig,
)
from zenml.logger import get_logger
from zenml import ArtifactConfig
from zenml import log_artifact_metadata

logger = get_logger(__name__)


@step
def deploy_model_to_hf_hub(
    endpoint_name: str,
    framework: str,
    task: str,
    accelerator: str,
    vendor: str,
    region: str,
    endpoint_type: str,
    instance_size: str,
    instance_type: str,
) -> Annotated[
    HuggingFaceModelService,
    ArtifactConfig(name="endpoint", is_deployment_artifact=True),
]:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        endpoint_name (str): The name of the endpoint to create on huggingface.
        repository (str): The repository on huggingface.
        framework (str): The machine learning framework used.
        task (str): The task that the machine learning model should perform.
        accelerator (str): Type of accelerator to use.
        vendor (str): The cloud service provider.
        region (str): The cloud region where the service will be hosted.
        endpoint_type (str): The type of the endpoint.
        instance_size (str): The size of the instance.
        instance_type (str): The type of the instance to use.

    """
    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]

    revision = get_step_context().model_version.metadata["revision"]
    repository = get_step_context().model_version.metadata["repository"]
    logger.info(revision)
    logger.info(repository)

    hf_endpoint_cfg = HFInferenceEndpointConfig(
        endpoint_name=endpoint_name,
        revision=revision,
        repository=repository,
        framework=framework,
        task=task,
        accelerator=accelerator,
        vendor=vendor,
        region=region,
        endpoint_type=endpoint_type,
        instance_size=instance_size,
        instance_type=instance_type,
        hf_token=hf_token,
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
