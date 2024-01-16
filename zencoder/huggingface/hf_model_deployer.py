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
from zenml.model_deployers.base_model_deployer import BaseModelDeployerFlavor

logger = get_logger(__name__)


class HFEndpointModelDeployer(BaseModelDeployer):
    NAME: ClassVar[str] = "HFEndpoint"
    FLAVOR: ClassVar[Type[BaseModelDeployerFlavor]] = HFModelDeployerFlavor

    def deploy_model(
        self, config: ServiceConfig, replace: bool = False, timeout: int = 300
    ) -> BaseService:
        """_summary_.

        Args:
            config (ServiceConfig): _description_
            replace (bool, optional): _description_. Defaults to False.
            timeout (int, optional): _description_. Defaults to 300.

        Raises:
            ValueError: _description_

        Returns:
            BaseService: _description_
        """
        config = cast(HFInferenceEndpointConfig, config)

        if config.revision is None or config.repository is None:
            raise ValueError(
                "The ZenML model version does not have a repository or revision in its metadata. "
                "Please make sure that the training pipeline is configured correctly."
            )

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

        # start the service which in turn provisions the Seldon Core
        # deployment server and waits for it to reach a ready state
        service.start(timeout=timeout)

        return service

    # TODO: Override rest of the abstract classes
