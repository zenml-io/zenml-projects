from zenml.logger import get_logger
import time
from typing import Generator, Tuple, Optional, Any, Dict
from zenml.services import ServiceType, ServiceState, ServiceStatus
from zenml.services.service import BaseDeploymentService, ServiceConfig
from huggingface_hub import (
    InferenceClient,
    InferenceEndpointError,
)
from huggingface_hub import create_inference_endpoint, get_inference_endpoint

from pydantic import Field

logger = get_logger(__name__)
POLLING_TIMEOUT = 1200


class HFInferenceEndpointConfig(ServiceConfig):
    """Base class for all ZenML model deployer configurations."""

    endpoint_name: str
    repository: str
    framework: str
    accelerator: str
    instance_size: str
    instance_type: str
    region: str
    vendor: str
    hf_token: str
    account_id: Optional[str] = None
    min_replica: Optional[int] = 0
    max_replica: Optional[int] = 1
    revision: Optional[str] = None
    task: Optional[str] = None
    custom_image: Optional[Dict] = None
    namespace: Optional[str] = None
    endpoint_type: str = "public"


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

    def wait_for_startup(self, timeout: int = POLLING_TIMEOUT) -> bool:
        """_summary_

        Args:
            timeout (int, optional): _description_. Defaults to POLLING_TIMEOUT.

        Returns:
            bool: _description_
        """
        # Wait for initialization
        try:
            # Add timelimit
            start_time = time.time()
            endpoint = None
            while endpoint is None:
                logger.info(
                    f"Waiting for {self.config.endpoint_name} to deploy. This might take a few minutes.."
                )
                endpoint = get_inference_endpoint(
                    name=self.config.endpoint_name,
                    token=self.config.hf_token,
                    namespace=self.config.namespace,
                )
                time.sleep(5)

                if time.time() - start_time > timeout:
                    return False

            self.endpoint = endpoint
            logger.info(f"Endpoint: {self.endpoint}")
            return True

        except KeyboardInterrupt:
            logger.info("Detected keyboard interrupt. Stopping polling.")
        return False

    def provision(self) -> None:
        """_summary_."""

        self.endpoint = create_inference_endpoint(
            name=self.config.endpoint_name,
            repository=self.config.repository,
            framework=self.config.framework,
            accelerator=self.config.accelerator,
            instance_size=self.config.instance_size,
            instance_type=self.config.instance_type,
            region=self.config.region,
            vendor=self.config.vendor,
            account_id=self.config.account_id,
            min_replica=self.config.min_replica,
            max_replica=self.config.max_replica,
            revision=self.config.revision,
            task=self.config.task,
            custom_image=self.config.custom_image,
            type=self.config.endpoint_type,
            namespace=self.config.namespace,
            token=self.config.hf_token,
        )
        logger.info(f"Endpoint: {self.endpoint}")

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
        return self.endpoint.client

    def check_status(self) -> Tuple[ServiceState, str]:
        """_summary_.

        Returns:
            Tuple[ServiceState, str]: _description_
        """
        try:
            client = self._get_client()
        except InferenceEndpointError:
            return (ServiceState.INACTIVE, "")

        if self.endpoint.status == "running":
            return (
                ServiceState.ACTIVE,
                f"HuggingFace Inference Endpoint deployment is available",
            )

        if self.endpoint.status == "failed":
            return (
                ServiceState.ERROR,
                f"HuggingFace Inference Endpoint deployment failed: ",
            )

        if self.endpoint.status == "pending":
            return (
                ServiceState.PENDING_STARTUP,
                "HuggingFace Inference Endpoint deployment is being created: ",
            )

    def deprovision(self, force: bool = False) -> None:
        """_summary_.

        Args:
            force (bool, optional): _description_. Defaults to False.
        """
        self.endpoint.delete()

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
        if self.endpoint.prediction_url is not None:
            client = self._get_client()
            if self.endpoint.task == "text-generation":
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
        return self.endpoint.url

    def get_logs(
        self, follow: bool = False, tail: int = None
    ) -> Generator[str, bool, None]:
        return super().get_logs(follow, tail)
