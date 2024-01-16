from zenml.model_deployers import BaseModelDeployer
from huggingface.hf_model_deployer_flavor import (
    HFInferenceEndpointConfig,
    HFModelDeployerFlavor,
)
from zenml.logger import get_logger

from typing import cast, ClassVar, Type
from zenml.services import BaseService, ServiceConfig
from huggingface_hub import create_inference_endpoint
from huggingface.hf_deployment import HuggingFaceModelService
from zenml.model_deployers.base_model_deployer import (
    DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
    BaseModelDeployerFlavor,
)

logger = get_logger(__name__)


class HFEndpointModelDeployer(BaseModelDeployer):
    """Huggingface endpoint model deployer."""

    NAME: ClassVar[str] = "HFEndpoint"
    FLAVOR: ClassVar[Type[BaseModelDeployerFlavor]] = HFModelDeployerFlavor

    def deploy_model(
        self,
        config: ServiceConfig,
        replace: bool = False,
        timeout: int = DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
    ) -> BaseService:
        """_summary_.

        Args:
            config (ServiceConfig): _description_
            replace (bool, optional): _description_. Defaults to False.
            timeout (int, optional): _description_. Defaults to DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT.

        Raises:
            ValueError: _description_

        Returns:
            BaseService: _description_
        """
        config = cast(HFInferenceEndpointConfig, config)

        endpoint = create_inference_endpoint(
            endpoint_name=config.endpoint_name,
            repository=config.repository,
            revision=config.revision,
            framework=config.framework,
            task=config.task,
            accelerator=config.accelerator,
            vendor=config.vendor,
            region=config.region,
            type=config.type,
            instance_size=config.instance_size,
            instance_type=config.instance_type,
            token=config.hf_token,
        )

        service = HuggingFaceModelService(endpoint=endpoint)

        logger.info(
            f"Creating a new huggingface inference endpoint service: {service}"
        )

        # start the service
        # and wait for it to reach a ready state
        service.start(timeout=timeout)

        return service

    # TODO: Override rest of the abstract classes
