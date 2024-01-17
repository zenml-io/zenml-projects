from uuid import UUID
from zenml.model_deployers import BaseModelDeployer
from huggingface.hf_model_deployer_flavor import HFModelDeployerFlavor
from zenml.logger import get_logger

from typing import List, Optional, cast, ClassVar, Type, Dict
from zenml.services import BaseService, ServiceConfig
from huggingface.hf_deployment import (
    HuggingFaceModelService,
    HFInferenceEndpointConfig,
)
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
        service = self._create_new_service(timeout, config)
        logger.info(
            f"Creating a new huggingface inference endpoint service: {service}"
        )
        return cast(BaseService, service)

    def _create_new_service(
        self, timeout: int, config: HFInferenceEndpointConfig
    ) -> HuggingFaceModelService:
        # create a new service for the new model
        service = HuggingFaceModelService(config)
        service.start(timeout=timeout)
        return service

    def find_model_server(
        self,
        running: bool,
        service_uuid: UUID,
        pipeline_name: str,
        run_name: str,
        pipeline_step_name: str,
        model_name: str,
        model_uri: str,
        model_type: str,
    ) -> List[BaseService]:
        pass

    def start_model_server(self, uuid: UUID, timeout: int = ...) -> None:
        raise NotImplementedError("Starting servers is not implemented")

    def stop_model_server(
        self, uuid: UUID, timeout: int = ..., force: bool = False
    ) -> None:
        raise NotImplementedError("Stopping servers is not implemented")

    def delete_model_server(
        self, uuid: UUID, timeout: int = ..., force: bool = False
    ) -> None:
        raise NotImplementedError("Deleting servers is not implemented")

    def get_model_server_info(
        service_instance: "HuggingFaceModelService",
    ) -> Dict[str, Optional[str]]:
        return {
            "PREDICTION_URL": service_instance.prediction_url,
        }
