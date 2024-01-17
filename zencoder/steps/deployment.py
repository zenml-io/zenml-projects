from zenml import step
from zenml.client import Client
from huggingface_hub import create_inference_endpoint
from zenml import ArtifactConfig
from typing_extensions import Annotated
from zenml import get_step_context
from typing import Optional, Dict
import random
from zenml import log_artifact_metadata
from zenml.metadata.metadata_types import Uri

def generate_three_random_letters() -> str:
    """Generates three random letters.

    Returns:
        Three random letters.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(letters) for i in range(3))


@step
def deploy_model_to_hf_hub(
    framework: str,
    accelerator: str,
    instance_size: str,
    instance_type: str,
    region: str,
    vendor: str,
    name: Optional[str] = None,
    account_id: Optional[str] = None,
    min_replica: int = 0,
    max_replica: int = 1,
    task: Optional[str] = None,
    custom_image: Optional[Dict] = None,
    endpoint_type: str = "protected",
) -> Annotated[str, ArtifactConfig(name="huggingface_service", is_deployment_artifact=True)]:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        framework: The framework of the model.
        accelerator: The accelerator of the model.
        instance_size: The instance size of the model.
        instance_type: The instance type of the model.
        region: The region of the model.
        vendor: The vendor of the model.
        name: The name of the model.
        account_id: The account id of the model.
        min_replica: The minimum replica of the model.
        max_replica: The maximum replica of the model.
        task: The task of the model.
        custom_image: The custom image of the model.
        endpoint_type: The type of the model.
    """
    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]
    
    revision = get_step_context().model_version.metadata["revision"]
    repository = get_step_context().model_version.metadata["repository"]
    namespace = ""

    if repository is None:
        raise ValueError(
            "The ZenML model version does not have a repository in its metadata. "
            "Please make sure that the training pipeline is configured correctly."
        )

    if name is None:
        name = repository + "-" + generate_three_random_letters()

    endpoint = create_inference_endpoint(
        name=name,
        repository=repository,
        framework=framework,
        accelerator=accelerator,
        instance_size=instance_size,
        instance_type=instance_type,
        region=region,
        vendor=vendor,
        account_id=account_id,
        min_replica=min_replica,
        max_replica=max_replica,
        revision=revision,
        task=task,
        custom_image=custom_image,
        type=endpoint_type,
        namespace=namespace,
        token=hf_token,
    )
    
    model_url = f"https://huggingface.co/{namespace}/{repository}"
    if  revision:
        model_url = f"{model_url}/commit/{revision}"

    log_artifact_metadata(
        metadata={
            "service_type": "huggingface",
            "status": "active",
            "description": "Huggingface Inference Endpoint",
            "endpoint_url": Uri(endpoint),
            "huggingface_model": Uri(model_url)
        }
    )
    
    return endpoint
