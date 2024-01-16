from huggingface_hub import HfApi
from zenml import step
from zenml.client import Client
from huggingface_hub import create_inference_endpoint
from zenml import ArtifactConfig
from typing_extensions import Annotated
from zenml import get_step_context

@step
def deploy_model(
    endpoint_name: str,
    framework: str,
    task: str,
    accelerator: str,
    vendor: str,
    region: str,
    type: str,
    instance_size: str,
    instance_type: str,
) -> Annotated[str, ArtifactConfig(name="endpoint", is_deployment_artifact=True)]:
    """Pushes the dataset to the Hugging Face Hub.

    Args:
        endpoint_name (str): The name of the endpoint to create on huggingface.
        repository (str): The repository on huggingface.
        framework (str): The machine learning framework used.
        task (str): The task that the machine learning model should perform.
        accelerator (str): Type of accelerator to use.
        vendor (str): The cloud service provider.
        region (str): The cloud region where the service will be hosted.
        type (str): The type of the endpoint.
        instance_size (str): The size of the instance.
        instance_type (str): The type of the instance to use.

    """
    secret = Client().get_secret("huggingface_creds")
    hf_token = secret.secret_values["token"]
    
    revision = get_step_context().model_version.metadata["revision"]
    repository = get_step_context().model_version.metadata["repository"]

    if revision is None or repository is None:
        raise ValueError(
            "The ZenML model version does not have a repository or revision in its metadata. "
            "Please make sure that the training pipeline is configured correctly."
        )

    endpoint = create_inference_endpoint(
        endpoint_name=endpoint_name,
        repository=repository,
        revision=revision,
        framework=framework,
        task=task,
        accelerator=accelerator,
        vendor=vendor,
        region=region,
        type=type,
        instance_size=instance_size,
        instance_type=instance_type,
        token=hf_token
    )
    return endpoint
