from zenml.logger import get_logger
from typing import Generator, Tuple, Optional, Any, List
from zenml.services import ServiceType, ServiceState, ServiceStatus
from zenml.services.service import BaseDeploymentService, ServiceConfig
from huggingface_hub import (
    InferenceClient,
    InferenceEndpointError,
    InferenceEndpoint,
)
from huggingface_hub import (
    create_inference_endpoint,
    get_inference_endpoint,
    list_inference_endpoints,
)
from huggingface.hf_deployment_base_config import HuggingFaceBaseConfig

from pydantic import Field

logger = get_logger(__name__)

POLLING_TIMEOUT = 1200


class HuggingFaceServiceConfig(HuggingFaceBaseConfig, ServiceConfig):
    """Base class for Huggingface configurations."""

    def get_deployed_endpoints(self) -> List[InferenceEndpoint]:
        """_summary_

        Returns:
            List[InferenceEndpoint]: _description_
        """
        return list_inference_endpoints(
            token=self.config.token,
            namespace=self.config.namespace,
        )


class HuggingFaceServiceStatus(ServiceStatus):
    """HF Endpoint Inference service status."""


class HuggingFaceDeploymentService(BaseDeploymentService):
    """HuggingFace model deployment service."""

    SERVICE_TYPE = ServiceType(
        name="hf-endpoint-deployment",
        type="model-serving",
        flavor="hfendpoint",
        description="Huggingface inference endpoint service",
    )
    config: HuggingFaceServiceConfig
    status: HuggingFaceServiceStatus = Field(
        default_factory=lambda: HuggingFaceServiceStatus()
    )

    def __init__(self, config: HuggingFaceServiceConfig, **attrs: Any):
        """_summary_."""
        super().__init__(config=config, **attrs)

    @property
    def hf_endpoint(self) -> InferenceEndpoint:
        """_summary_

        Returns:
            InferenceEndpoint: _description_
        """
        return get_inference_endpoint(
            name=self.config.endpoint_name,
            token=self.config.token,
            namespace=self.config.namespace,
        )

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

    @property
    def inference_client(self) -> InferenceClient:
        """_summary_.

        Returns:
            InferenceClient: _description_
        """
        return self.hf_endpoint.client

    def provision(self) -> None:
        """_summary_."""

        _ = create_inference_endpoint(
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
            token=self.config.token,
        ).wait(timeout=POLLING_TIMEOUT)

        if self.hf_endpoint.url is not None:
            logger.info(
                "Huggingface inference endpoint successfully deployed."
            )
        else:
            logger.info(
                "Failed to start huggingface inference endpoint service..."
            )

    def check_status(self) -> Tuple[ServiceState, str]:
        """_summary_.

        Returns:
            Tuple[ServiceState, str]: _description_
        """
        try:
            _ = self.inference_client
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
            if self.hf_endpoint.task == "text-generation":
                result = self.inference_client.task_generation(
                    data, max_new_tokens=max_new_tokens
                )
        else:
            raise NotImplementedError(
                "Tasks other than text-generation is not implemented."
            )
        return result

    def get_logs(
        self, follow: bool = False, tail: int = None
    ) -> Generator[str, bool, None]:
        return super().get_logs(follow, tail)