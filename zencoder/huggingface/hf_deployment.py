from zenml.logger import get_logger
import time
from typing import Tuple, Optional
from zenml.services import ServiceType, ServiceState
from zenml.services.service import BaseDeploymentService
from huggingface_hub import (
    InferenceClient,
    InferenceEndpointError,
    InferenceEndpoint,
)

logger = get_logger(__name__)


class HuggingFaceModelService(BaseDeploymentService):
    """HuggingFace model deployment service."""

    SERVICE_TYPE = ServiceType(
        name="hf-endpoint-deployment",
        type="model-serving",
        flavor="hfendpoint",
        description="Huggingface inference endpoint service",
    )

    def __init__(self, endpoint: InferenceEndpoint):
        """_summary_.

        Args:
            endpoint (InferenceEndpoint): _description_
        """
        self.endpoint = endpoint

    def wait_for_startup(self, timeout: int = 300) -> bool:
        """_summary_

        Args:
            timeout (int, optional): _description_. Defaults to 300.

        Returns:
            bool: _description_
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.endpoint.status == "running":
                return True
            time.sleep(5)  # Adjust the sleep interval as needed

        return False

    def provision(self) -> None:
        """_summary_."""
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
                "BentoML prediction service is not running. "
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
