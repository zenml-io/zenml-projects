from uuid import UUID
from zenml.model_deployers import BaseModelDeployer
from huggingface.hf_model_deployer_flavor import HuggingFaceModelDeployerFlavor
from zenml.logger import get_logger

from typing import List, Optional, cast, ClassVar, Type, Dict
from zenml.services import BaseService, ServiceConfig
from huggingface.hf_deployment_service import (
    HuggingFaceDeploymentService,
    HuggingFaceServiceConfig,
)
from huggingface.hf_model_deployer_flavor import (
    HuggingFaceModelDeployerSettings,
    HuggingFaceModelDeployerConfig,
)
from zenml.model_deployers.base_model_deployer import (
    DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
    BaseModelDeployerFlavor,
)
from zenml.client import Client
from zenml.services import ServiceRegistry
from zenml.artifacts.utils import save_artifact, log_artifact_metadata

logger = get_logger(__name__)


class HuggingFaceModelDeployer(BaseModelDeployer):
    """Huggingface endpoint model deployer."""

    NAME: ClassVar[str] = "HFEndpoint"
    FLAVOR: ClassVar[
        Type[BaseModelDeployerFlavor]
    ] = HuggingFaceModelDeployerFlavor

    @property
    def config(self) -> HuggingFaceModelDeployerConfig:
        """Config class for the Huggingface Model deployer settings class.

        Returns:
            The configuration.
        """
        return cast(HuggingFaceModelDeployerConfig, self._config)

    @property
    def settings_class(self) -> Type[HuggingFaceModelDeployerSettings]:
        """Settings class for the Huggingface Model deployer settings class.

        Returns:
            The settings class.
        """
        return HuggingFaceModelDeployerSettings

    def _create_new_service(
        self, timeout: int, config: HuggingFaceServiceConfig
    ) -> HuggingFaceDeploymentService:
        """Creates a new HuggingFaceDeploymentService.

        Args:
            timeout: the timeout in seconds to wait for the Huggingface inference endpoint
                to be provisioned and successfully started or updated.
            config: the configuration of the model to be deployed with Huggingface model deployer.

        Returns:
            The HuggingFaceServiceConfig object that can be used to interact
            with the Huggingface inference endpoint.
        """
        # create a new service for the new model
        service = HuggingFaceDeploymentService(config)

        service_metadata = service.dict()

        save_artifact(
            service,
            "hf_deployment_service",
            version=str(service_metadata["uuid"]),
            is_deployment_artifact=True,
        )
        # UUID object is not json serializable
        service_metadata["uuid"] = str(service_metadata["uuid"])
        log_artifact_metadata(
            metadata={"hf_deployment_service": service_metadata}
        )

        service.start(timeout=timeout)
        return service

    def _clean_up_existing_service(
        self,
        timeout: int,
        force: bool,
        existing_service: HuggingFaceDeploymentService,
    ) -> None:
        """_summary_

        Args:
            timeout (int): _description_
            force (bool): _description_
            existing_service (HuggingFaceDeploymentService): _description_
        """
        # stop the older service
        existing_service.stop(timeout=timeout, force=force)

    def deploy_model(
        self,
        config: ServiceConfig,
        replace: bool = False,
        timeout: int = DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
    ) -> BaseService:
        """Create a new Huggingface deployment service or update an existing one.

        This should serve the supplied model and deployment configuration.

        Args:
            config (ServiceConfig): _description_
            replace (bool, optional): _description_. Defaults to False.
            timeout (int, optional): _description_. Defaults to DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT.

        Raises:
            ValueError: _description_

        Returns:
            BaseService: _description_
        """
        config = cast(HuggingFaceServiceConfig, config)
        service = None

        # Add zenml prefix
        if not config.endpoint_name.startswith("zenml-"):
            config.endpoint_name = "zenml-" + config.endpoint_name

        # if replace is True, remove all existing services
        if replace is True:
            existing_services = self.find_model_server(
                pipeline_name=config.pipeline_name,
                pipeline_step_name=config.pipeline_step_name,
            )

            for existing_service in existing_services:
                if service is None:
                    # keep the most recently created service
                    service = cast(
                        HuggingFaceDeploymentService, existing_service
                    )
                try:
                    # delete the older services and don't wait for them to
                    # be deprovisioned
                    self._clean_up_existing_service(
                        existing_service=cast(
                            HuggingFaceDeploymentService, existing_service
                        ),
                        timeout=timeout,
                        force=True,
                    )
                except RuntimeError:
                    # ignore errors encountered while stopping old services
                    pass

        if service:
            logger.info(
                f"Updating an existing Hugginface deployment service: {service}"
            )
            service.stop(timeout=timeout, force=True)
            service.update(config)
            service.start(timeout=timeout)
        else:
            # create a new HuggingFaceDeploymentService instance
            service = self._create_new_service(timeout, config)
            logger.info(
                f"Creating a new huggingface inference endpoint service: {service}"
            )

        return cast(BaseService, service)

    def find_model_server(
        self,
        running: bool = False,
        service_uuid: Optional[UUID] = None,
        pipeline_name: Optional[str] = None,
        run_name: Optional[str] = None,
        pipeline_step_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_uri: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> List[BaseService]:
        # Use a Huggingface deployment service configuration to compute the labels
        config = HuggingFaceServiceConfig(
            pipeline_name=pipeline_name or "",
            run_name=run_name or "",
            pipeline_run_id=run_name or "",
            pipeline_step_name=pipeline_step_name or "",
            model_name=model_name or "",
            model_uri=model_uri or "",
            implementation=model_type or "",
        )

        # List all the inference endpoints
        endpoints = config.get_deployed_endpoints()

        services: List[BaseService] = []
        for endpoint in endpoints:
            if endpoint.name.startswith("zenml-"):
                client = Client()
                service_artifact = client.get_artifact_version(
                    "hf_deployment_service", str(service_uuid)
                )
                hf_deployment_service_dict = service_artifact.metadata[
                    "hf_deployment_service"
                ]
                existing_service = ServiceRegistry().load_service_from_dict(
                    hf_deployment_service_dict
                )

                if not isinstance(
                    existing_service, HuggingFaceDeploymentService
                ):
                    raise TypeError(
                        f"Expected service type HuggingFaceDeploymentService but got "
                        f"{type(existing_service)} instead"
                    )

                existing_service.update_status()
                if self._matches_search_criteria(existing_service, config):
                    if not running or existing_service.is_running:
                        services.append(cast(BaseService, existing_service))

            services.append(existing_service)

        return services

    def _matches_search_criteria(
        self,
        existing_service: HuggingFaceDeploymentService,
        config: HuggingFaceModelDeployerConfig,
    ) -> bool:
        """Returns true if a service matches the input criteria.

        If any of the values in the input criteria are None, they are ignored.
        This allows listing services just by common pipeline names or step
        names, etc.

        Args:
            existing_service: The materialized Service instance derived from
                the config of the older (existing) service
            config: The BentoMlDeploymentConfig object passed to the
                deploy_model function holding parameters of the new service
                to be created.

        Returns:
            True if the service matches the input criteria.
        """
        existing_service_config = existing_service.config

        # check if the existing service matches the input criteria
        if (
            (
                not config.pipeline_name
                or existing_service_config.pipeline_name
                == config.pipeline_name
            )
            and (
                not config.model_name
                or existing_service_config.model_name == config.model_name
            )
            and (
                not config.pipeline_step_name
                or existing_service_config.pipeline_step_name
                == config.pipeline_step_name
            )
            and (
                not config.run_name
                or existing_service_config.run_name == config.run_name
            )
        ):
            return True

        return False

    def stop_model_server(
        self,
        uuid: UUID,
        timeout: int = DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
        force: bool = False,
    ) -> None:
        """Method to stop a model server.

        Args:
            uuid: UUID of the model server to stop.
            timeout: Timeout in seconds to wait for the service to stop.
            force: If True, force the service to stop.
        """
        # get list of all services
        existing_services = self.find_model_server(service_uuid=uuid)

        # if the service exists, stop it
        if existing_services:
            existing_services[0].stop(timeout=timeout, force=force)

    def start_model_server(
        self, uuid: UUID, timeout: int = DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT
    ) -> None:
        """Method to start a model server.

        Args:
            uuid: UUID of the model server to start.
            timeout: Timeout in seconds to wait for the service to start.
        """
        # get list of all services
        existing_services = self.find_model_server(service_uuid=uuid)

        # if the service exists, start it
        if existing_services:
            existing_services[0].start(timeout=timeout)

    def delete_model_server(
        self,
        uuid: UUID,
        timeout: int = DEFAULT_DEPLOYMENT_START_STOP_TIMEOUT,
        force: bool = False,
    ) -> None:
        """Method to delete all configuration of a model server.

        Args:
            uuid: UUID of the model server to delete.
            timeout: Timeout in seconds to wait for the service to stop.
            force: If True, force the service to stop.
        """
        # get list of all services
        existing_services = self.find_model_server(service_uuid=uuid)

        # if the service exists, clean it up
        if existing_services:
            service = cast(HuggingFaceDeploymentService, existing_services[0])
            self._clean_up_existing_service(
                existing_service=service, timeout=timeout, force=force
            )

    def get_model_server_info(
        service_instance: "HuggingFaceDeploymentService",
    ) -> Dict[str, Optional[str]]:
        return {
            "PREDICTION_URL": service_instance.prediction_url,
        }