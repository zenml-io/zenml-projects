#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Implementation of the Seldon Model Deployer."""

from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Type, cast
from uuid import UUID

import boto3
import sagemaker
from hf_sagemaker_client import (
    HFSagemakerClient,
)
from hf_sagemaker_deployment_service import (
    HFSagemakerDeploymentConfig,
    HFSagemakerDeploymentService,
)
from hf_sagemaker_model_deployer_flavor import (
    HFSagemakerModelDeployerConfig,
    HFSagemakerModelDeployerFlavor,
)
from zenml.logger import get_logger
from zenml.model_deployers import BaseModelDeployer, BaseModelDeployerFlavor
from zenml.services.service import BaseService

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


DEFAULT_HUGGINGFACE_SAGEMAKER_DEPLOYMENT_START_STOP_TIMEOUT = 300
ENV_AWS_PROFILE = "AWS_PROFILE"


class HFSagemakerModelDeployer(BaseModelDeployer):
    """Huggingface Sagemaker model deployer stack component implementation."""

    NAME: ClassVar[str] = "Huggingface Sagemaker"
    FLAVOR: ClassVar[Type[BaseModelDeployerFlavor]] = HFSagemakerModelDeployerFlavor

    @property
    def config(self) -> HFSagemakerModelDeployerConfig:
        """Returns the `HFSagemakerModelDeployerConfig` config.

        Returns:
            The configuration.
        """
        return cast(HFSagemakerModelDeployerConfig, self._config)

    @property
    def hf_sagemaker_client(self) -> HFSagemakerClient:
        """Get the Seldon Core client associated with this model deployer.

        Returns:
            The Seldon Core client.

        Raises:
            RuntimeError: If the Kubernetes namespace is not configured when
                using a service connector to deploy models with Seldon Core.
        """
        self._client = HFSagemakerClient(
            sagemaker_session=self.get_sagemaker_session(self.config),
        )
        return self._client

    def get_sagemaker_session(
        self, config: HFSagemakerDeploymentConfig
    ) -> sagemaker.Session:
        """Returns sagemaker session from connector"""
        # Refresh the client also if the connector has expired
        if self._client and not self.connector_has_expired():
            return self._client

        # Get authenticated session
        # Option 1: Service connector
        boto_session: boto3.Session
        if connector := self.get_connector():
            boto_session = connector.connect()
            if not isinstance(boto_session, boto3.Session):
                raise RuntimeError(
                    f"Expected to receive a `boto3.Session` object from the "
                    f"linked connector, but got type `{type(boto_session)}`."
                )
        # Option 2: Explicit configuration
        elif config.sagemaker_session_args:
            boto_session = boto3.Session(**config.sagemaker_session_args)
        # Option 3: Implicit configuration
        else:
            boto_session = boto3.Session()

        return sagemaker.Session(boto_session=boto_session)

    @staticmethod
    def get_model_server_info(  # type: ignore[override]
        service_instance: "HFSagemakerDeploymentService",
    ) -> Dict[str, Optional[str]]:
        """Return implementation specific information that might be relevant to the user.

        Args:
            service_instance: Instance of a HFSagemakerDeploymentService

        Returns:
            Model server information.
        """
        return {
            "ENDPOINT_NAME": service_instance.endpoint_name,
        }

    def deploy_model(
        self,
        config: HFSagemakerDeploymentConfig,
        replace: bool = False,
        timeout: int = DEFAULT_HUGGINGFACE_SAGEMAKER_DEPLOYMENT_START_STOP_TIMEOUT,
    ) -> BaseService:
        """Create a new Sagemaker Huggingface deployment or update an existing one."""
        config = cast(HFSagemakerDeploymentConfig, config)
        service = None

        # sagemaker_session = self.get_sagemaker_session(config)
        # iam_role_arn = config.iam_role_arn or self.config.iam_role_arn
        # if not iam_role_arn:
        #    raise ValueError(
        #        "No IAM role ARN was provided. Please provide one either "
        #        "through the connector or through the config."
        #    )
        # if replace is True, find equivalent deployments
        if replace is True:
            equivalent_services = self.find_model_server(
                running=False,
                pipeline_name=config.pipeline_name,
                pipeline_step_name=config.pipeline_step_name,
                model_name=config.model_name,
            )

            for equivalent_service in equivalent_services:
                if service is None:
                    # keep the most recently created service
                    service = equivalent_service
                else:
                    try:
                        # delete the older services and don't wait for
                        # them to be deprovisioned
                        service.stop()
                    except RuntimeError:
                        # ignore errors encountered while stopping old
                        # services
                        pass

        if service:
            # update an equivalent service in place
            service.update(config)
            logger.info(f"Updating an existing Seldon deployment service: {service}")
        else:
            # create a new service
            service = HFSagemakerDeploymentService(config=config)
            logger.info(
                f"Creating a new Sagemaker Huggingface deployment service: {service}"
            )

        # start the service which in turn provisions the Seldon Core
        # deployment server and waits for it to reach a ready state
        service.start(timeout=timeout)

        # Add telemetry with metadata that gets the stack metadata and
        # differentiates between pure model and custom code deployments

        return service

    def find_model_server(
        self,
        running: bool = False,
        service_uuid: Optional[UUID] = None,
        pipeline_name: Optional[str] = None,
        run_name: Optional[str] = None,
        pipeline_step_name: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        image_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[BaseService]:
        """Find one or more Seldon Core model services that match the given criteria.

        The Seldon Core deployment services that meet the search criteria are
        returned sorted in descending order of their creation time (i.e. more
        recent deployments first).

        Args:
            running: if true, only running services will be returned.
            service_uuid: the UUID of the Seldon Core service that was
                originally used to create the Seldon Core deployment resource.
            pipeline_name: name of the pipeline that the deployed model was part
                of.
            run_name: Name of the pipeline run which the deployed model was
                part of.
            pipeline_step_name: the name of the pipeline model deployment step
                that deployed the model.
            model_name: the name of the deployed model.
            model_uri: URI of the deployed model.
            model_type: the Seldon Core server implementation used to serve
                the model

        Returns:
            One or more Seldon Core service objects representing Seldon Core
            model servers that match the input search criteria.
        """
        # Use a Seldon deployment service configuration to compute the labels
        config = HFSagemakerDeploymentConfig(
            pipeline_name=pipeline_name or "",
            run_name=run_name or "",
            pipeline_run_id=run_name or "",
            pipeline_step_name=pipeline_step_name or "",
            endpoint_name=endpoint_name or "",
            image_uri=image_uri or "",
        )
        tags = config.get_hf_sagemaker_deployment_tags()
        # if service_uuid:
        # the service UUID is not a label covered by the Seldon
        # deployment service configuration, so we need to add it
        # separately
        #    tags["zenml.service_uuid"] = str(service_uuid)

        deployments = self.hf_sagemaker_client.find_deployments(tags=tags)
        # sort the deployments in descending order of their creation time

        services: List[BaseService] = []
        for deployment in deployments:
            # recreate the Seldon deployment service object from the Seldon
            # deployment resource
            service = HFSagemakerDeploymentService.create_from_deployment(
                deployment=deployment
            )
            if running and not service.is_running:
                # skip non-running services
                continue
            services.append(service)

        return services

    def stop_model_server(
        self,
        uuid: UUID,
        timeout: int = DEFAULT_SELDON_DEPLOYMENT_START_STOP_TIMEOUT,
        force: bool = False,
    ) -> None:
        """Stop a Seldon Core model server.

        Args:
            uuid: UUID of the model server to stop.
            timeout: timeout in seconds to wait for the service to stop.
            force: if True, force the service to stop.

        Raises:
            NotImplementedError: stopping Seldon Core model servers is not
                supported.
        """
        raise NotImplementedError(
            "Stopping Seldon Core model servers is not implemented. Try "
            "deleting the Seldon Core model server instead."
        )

    def start_model_server(
        self,
        uuid: UUID,
        timeout: int = DEFAULT_SELDON_DEPLOYMENT_START_STOP_TIMEOUT,
    ) -> None:
        """Start a Seldon Core model deployment server.

        Args:
            uuid: UUID of the model server to start.
            timeout: timeout in seconds to wait for the service to become
                active. . If set to 0, the method will return immediately after
                provisioning the service, without waiting for it to become
                active.

        Raises:
            NotImplementedError: since we don't support starting Seldon Core
                model servers
        """
        raise NotImplementedError(
            "Starting Seldon Core model servers is not implemented"
        )

    def delete_model_server(
        self,
        uuid: UUID,
        timeout: int = DEFAULT_SELDON_DEPLOYMENT_START_STOP_TIMEOUT,
        force: bool = False,
    ) -> None:
        """Delete a Seldon Core model deployment server.

        Args:
            uuid: UUID of the model server to delete.
            timeout: timeout in seconds to wait for the service to stop. If
                set to 0, the method will return immediately after
                deprovisioning the service, without waiting for it to stop.
            force: if True, force the service to stop.
        """
        services = self.find_model_server(service_uuid=uuid)
        if len(services) == 0:
            return

        service = services[0]

        assert isinstance(service, SeldonDeploymentService)
        service.stop(timeout=timeout, force=force)

        if service.config.secret_name:
            # delete the Kubernetes secret used to store the authentication
            # information for the Seldon Core model server storage initializer
            # if no other Seldon Core model servers are using it
            self._delete_kubernetes_secret(service.config.secret_name)
