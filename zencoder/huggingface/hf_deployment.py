from zenml.logger import get_logger
import time
from typing import Generator, Tuple, Optional, Any
from zenml.services import ServiceType, ServiceState, ServiceStatus
from zenml.services.service import BaseDeploymentService, ServiceConfig
from huggingface_hub import (
    InferenceClient,
    InferenceEndpointError,
)
from huggingface_hub import create_inference_endpoint

from pydantic import Field

logger = get_logger(__name__)


class HFInferenceEndpointConfig(ServiceConfig):
    """Base class for all ZenML model deployer configurations."""

    endpoint_name: str
    revision: str
    repository: str
    framework: str
    task: str
    accelerator: str
    vendor: str
    region: str
    endpoint_type: str
    instance_size: str
    instance_type: str
    hf_token: str


class HFEndpointServiceStatus(ServiceStatus):
    """HF Endpoint Inference service status."""


class HuggingFaceModelService(BaseDeploymentService):
    """HuggingFace model deployment service."""

    SERVICE_TYPE = ServiceType(
        name="hf-endpoint-deployment",
        type="model-serving",
        flavor="hfendpoint",
        description="Huggingface inference endpoint service",
    )
    config: HFInferenceEndpointConfig
    status: HFEndpointServiceStatus = Field(
        default_factory=lambda: HFEndpointServiceStatus()
    )

    def __init__(self, config: HFInferenceEndpointConfig, **attrs: Any):
        """_summary_.

        Args:
            endpoint (InferenceEndpoint): _description_
        """
        super().__init__(config=config, **attrs)

    def wait_for_startup(self, timeout: int = 300) -> bool:
        """_summary_

        Args:
            timeout (int, optional): _description_. Defaults to 300.

        Returns:
            bool: _description_
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.hf_endpoint.status == "running":
                return True
            time.sleep(5)  # Adjust the sleep interval as needed

        return False

    def provision(self) -> None:
        """_summary_."""

        self.hf_endpoint = create_inference_endpoint(
            name=self.config.endpoint_name,
            repository=self.config.repository,
            revision=self.config.revision,
            framework=self.config.framework,
            task=self.config.task,
            accelerator=self.config.accelerator,
            vendor=self.config.vendor,
            region=self.config.region,
            type=self.config.endpoint_type,
            instance_size=self.config.instance_size,
            instance_type=self.config.instance_type,
            token=self.config.hf_token,
        )

        if self.wait_for_startup():
            logger.info("Running huggingface inference endpoint.")
        else:
            logger.info(
                "Failed to start huggingface inference endpoint service..."
            )

    def _get_client(self) -> InferenceClient:
        """_summary_.

        Returns:
            InferenceClient: _description_
        """
        return self.hf_endpoint.client

    def check_status(self) -> Tuple[ServiceState, str]:
        """_summary_.

        Returns:
            Tuple[ServiceState, str]: _description_
        """
        try:
            client = self._get_client()
        except InferenceEndpointError:
            return (ServiceState.INACTIVE, "")

        if self.hf_endpoint.status == "running":
            return (
                ServiceState.ACTIVE,
                f"HuggingFace Inference Endpoint deployment is available",
            )

        if self.hf_endpoint.status == "failed":
            return (
                ServiceState.ERROR,
                f"HuggingFace Inference Endpoint deployment failed: ",
            )

        if self.hf_endpoint.status == "pending":
            return (
                ServiceState.PENDING_STARTUP,
                "HuggingFace Inference Endpoint deployment is being created: ",
            )

    def deprovision(self, force: bool = False) -> None:
        """_summary_.

        Args:
            force (bool, optional): _description_. Defaults to False.
        """
        self.hf_endpoint.delete()

    def predict(self, data: "Any", max_new_tokens: int) -> "Any":
        """Make a prediction using the service.

        Args:
            api_endpoint: the api endpoint to make the prediction on
            max_new_tokens: Number of new tokens to generate

        Returns:
            The prediction result.

        Raises:
            Exception: if the service is not running
            ValueError: if the prediction endpoint is unknown.
        """
        if not self.is_running:
            raise Exception(
                "Huggingface endpoint inference service is not running. "
                "Please start the service before making predictions."
            )
        if self.hf_endpoint.prediction_url is not None:
            client = self._get_client()
            if self.hf_endpoint.task == "text-generation":
                result = client.task_generation(
                    data, max_new_tokens=max_new_tokens
                )
        else:
            raise NotImplementedError(
                "Tasks other than text-generation is not implemented."
            )
        return result

    @property
    def prediction_url(self) -> Optional[str]:
        """The prediction URI exposed by the prediction service.

        Returns:
            The prediction URI exposed by the prediction service, or None if
            the service is not yet ready.
        """
        if not self.is_running:
            return None
        return self.hf_endpoint.url

    def get_logs(
        self, follow: bool = False, tail: int = None
    ) -> Generator[str, bool, None]:
        return super().get_logs(follow, tail)
