#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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
"""Implementation for the Sagemaker Huggingface Deployer step."""

import json
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, cast
from uuid import UUID

import boto3
import requests
import sagemaker
from pydantic import Field, ValidationError
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.huggingface.model import HuggingFacePredictor
from zenml import __version__
from zenml.logger import get_logger
from zenml.services.service import BaseDeploymentService, ServiceConfig
from zenml.services.service_status import ServiceState, ServiceStatus
from zenml.services.service_type import ServiceType

logger = get_logger(__name__)


class HFSagemakerDeploymentConfig(ServiceConfig):
    """"""

    # Huggingface model args
    iam_role_arn: Optional[str] = None
    model_data: Optional[str] = None
    entry_point: Optional[str] = None
    transformers_version: Optional[str] = None
    tensorflow_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    py_version: Optional[str] = None
    image_uri: Optional[str] = None
    model_server_workers: Optional[int] = None
    hf_model_uri: Optional[str] = None

    # Deploy args
    initial_instance_count: Optional[int] = None
    instance_type: Optional[str] = None
    accelerator_type: Optional[str] = None
    endpoint_name: Optional[str] = None
    tags: Optional[List[Dict[str, str]]] = []
    kms_key: Optional[str] = None
    wait: bool = True
    volume_size: Optional[int] = None
    model_data_download_timeout: Optional[int] = None
    container_startup_health_check_timeout: Optional[int] = None
    inference_recommendation_id: Optional[str] = None

    # Misc args
    env: Dict[str, str] = {}
    sagemaker_session_args: Dict[str, Any] = {}

    def get_hf_sagemaker_deployment_tags(self) -> Dict[str, str]:
        """Generate labels for the Seldon Core deployment from the service configuration.

        These labels are attached to the Seldon Core deployment resource
        and may be used as label selectors in lookup operations.

        Returns:
            The labels for the Seldon Core deployment.
        """
        if not tags:
            tags = {}
        else:
            for key, value in self.tags.items():
                tags.pop(key)
                tags[f"zenml.{key}"] = value
        tags["app"] = "zenml"
        if self.pipeline_name:
            tags["zenml.pipeline_name"] = self.pipeline_name
        if self.run_name:
            tags["zenml.run_name"] = self.run_name
        if self.pipeline_step_name:
            tags["zenml.pipeline_step_name"] = self.pipeline_step_name
        if self.model_name:
            tags["zenml.model_name"] = self.model_name
        if self.model_uri:
            tags["zenml.hf_model_uri"] = self.model_uri
        return tags

    def get_seldon_deployment_annotations(self) -> Dict[str, str]:
        """Generate annotations for the Seldon Core deployment from the service configuration.

        The annotations are used to store additional information about the
        Seldon Core service that is associated with the deployment that is
        not available in the labels. One annotation particularly important
        is the serialized Service configuration itself, which is used to
        recreate the service configuration from a remote Seldon deployment.

        Returns:
            The annotations for the Seldon Core deployment.
        """
        annotations = {
            "zenml.service_config": self.json(),
            "zenml.version": __version__,
        }
        return annotations

    @classmethod
    def create_from_deployment(
        cls, deployment: SeldonDeployment
    ) -> "HFSagemakerDeploymentConfig":
        """Recreate the configuration of a Seldon Core Service from a deployed instance.

        Args:
            deployment: the Seldon Core deployment resource.

        Returns:
            The Seldon Core service configuration corresponding to the given
            Seldon Core deployment resource.

        Raises:
            ValueError: if the given deployment resource does not contain
                the expected annotations or it contains an invalid or
                incompatible Seldon Core service configuration.
        """
        config_data = deployment.metadata.annotations.get("zenml.service_config")
        if not config_data:
            raise ValueError(
                f"The given deployment resource does not contain a "
                f"'zenml.service_config' annotation: {deployment}"
            )
        try:
            service_config = cls.parse_raw(config_data)
        except ValidationError as e:
            raise ValueError(
                f"The loaded Seldon Core deployment resource contains an "
                f"invalid or incompatible Seldon Core service configuration: "
                f"{config_data}"
            ) from e
        return service_config


class HFSagemakerDeploymentServiceStatus(ServiceStatus):
    """HF Sagemaker deployment service status."""


class HFSagemakerDeploymentService(BaseDeploymentService):
    """A service that represents a Seldon Core deployment server.

    Attributes:
        config: service configuration.
        status: service status.
    """

    SERVICE_TYPE = ServiceType(
        name="huggingface-sagemaker-deployment",
        type="model-serving",
        flavor="huggingface-sagemaker",
        description="Huggingface Sagemaker deployment service to deploy HF models on Sagemaker.",
    )

    config: HFSagemakerDeploymentConfig
    status: HFSagemakerDeploymentServiceStatus = Field(
        default_factory=lambda: HFSagemakerDeploymentServiceStatus()
    )

    def get_sagemaker_session(
        self, config: HFSagemakerDeploymentConfig
    ) -> sagemaker.Session:
        """Returns sagemaker session from connector"""
        session = sagemaker.Session(boto3.Session(**config.sagemaker_session_args))
        return session

    def check_status(self) -> Tuple[ServiceState, str]:
        """Check the the current operational state of the Seldon Core deployment.

        Returns:
            The operational state of the Seldon Core deployment and a message
            providing additional information about that state (e.g. a
            description of the error, if one is encountered).
        """
        client = self._get_client()
        name = self.seldon_deployment_name
        try:
            deployment = client.get_deployment(name=name)
        except SeldonDeploymentNotFoundError:
            return (ServiceState.INACTIVE, "")

        if deployment.is_available():
            return (
                ServiceState.ACTIVE,
                f"Seldon Core deployment '{name}' is available",
            )

        if deployment.is_failed():
            return (
                ServiceState.ERROR,
                f"Seldon Core deployment '{name}' failed: " f"{deployment.get_error()}",
            )

        pending_message = deployment.get_pending_message() or ""
        return (
            ServiceState.PENDING_STARTUP,
            "Seldon Core deployment is being created: " + pending_message,
        )

    @property
    def endpoint_name(self) -> str:
        """Get the name of the endpoint from sagemaker.

        It should return the one that uniquely corresponds to this service instance.

        Returns:
            The endpoint name of the deployed predictor.
        """
        return f"zenml-{str(self.uuid)}"

    def get_predictor(self) -> HuggingFacePredictor:
        return HuggingFacePredictor(self.config.endpoint_name)

    @classmethod
    def create_from_deployment(
        cls, deployment: sagemaker.Predictor
    ) -> "HFSagemakerDeploymentService":
        """Recreate a Seldon Core service from a Seldon Core deployment resource.

        It should then update their operational status.

        Args:
            deployment: the Seldon Core deployment resource.

        Returns:
            The Seldon Core service corresponding to the given
            Seldon Core deployment resource.

        Raises:
            ValueError: if the given deployment resource does not contain
                the expected service_uuid label.
        """
        config = HFSagemakerDeploymentConfig.create_from_deployment(deployment)
        uuid = deployment.metadata.labels.get("zenml.service_uuid")
        if not uuid:
            raise ValueError(
                f"The given deployment resource does not contain a valid "
                f"'zenml.service_uuid' label: {deployment}"
            )
        service = cls(uuid=UUID(uuid), config=config)
        service.update_status()
        return service

    def provision(self) -> None:
        """Provision or update remote Seldon Core deployment instance.

        This should then match the current configuration.
        """
        self.config["sagemaker_session"] = self.get_sagemaker_session

        # Hugging Face Model Class
        huggingface_model = HuggingFaceModel(
            env=self.config.env,
            role=self.config.role,
            model_data=self.config.model_data,
            entry_point=self.config.entry_point,
            transformers_version=self.config.transformers_version,
            tensorflow_version=self.config.tensorflow_version,
            pytorch_version=self.config.pytorch_version,
            py_version=self.config.py_version,
            image_uri=self.config.image_uri,
            model_server_workers=self.config.model_server_workers,
        )

        # Stamp this deployment with zenml metadata
        tags = self.config.tags
        annotations = {
            "zenml.service_config": self.json(),
            "zenml.version": __version__,
            "zenml.service_uuid": str(self.uuid),
        }
        """
        # TODO: Add these
        labels = {}
        if self.pipeline_name:
            labels["zenml.pipeline_name"] = self.pipeline_name
        if self.run_name:
            labels["zenml.run_name"] = self.run_name
        if self.pipeline_step_name:
            labels["zenml.pipeline_step_name"] = self.pipeline_step_name
        if self.model_name:
            labels["zenml.model_name"] = self.model_name
        if self.model_uri:
            labels["zenml.model_uri"] = self.model_uri
        if self.implementation:
            labels["zenml.model_type"] = self.implementation
        if self.extra_args:
            for key, value in self.extra_args.items():
                labels[f"zenml.{key}"] = value
        """

        tags.extend([annotations])

        # Override the endpoint name. This is critical to fetch it back
        endpoint_name = f"zenml_service_{str(self.uuid)}"

        # deploy model to SageMaker
        predictor: HuggingFacePredictor = huggingface_model.deploy(
            initial_instance_count=self.config.initial_instance_count,
            instance_type=self.config.instance_type,
            accelerator_type=self.config.accelerator_type,
            endpoint_name=endpoint_name,
            tags=tags,
            kms_key=self.config.kms_key,
            wait=self.config.wait,
            volume_size=self.config.volume_size,
            model_data_download_timeout=self.config.model_data_download_timeout,
            container_startup_health_check_timeout=self.config.container_startup_health_check_timeout,
            inference_recommendation_id=self.config.inference_recommendation_id,
        )
        predictor.endpoint_name

    def deprovision(self, force: bool = False) -> None:
        """Deprovision the remote Seldon Core deployment instance.

        Args:
            force: if True, the remote deployment instance will be
                forcefully deprovisioned.
        """
        client = self._get_client()
        name = self.seldon_deployment_name
        try:
            client.delete_deployment(name=name, force=force)
        except SeldonDeploymentNotFoundError:
            pass

    def get_logs(
        self,
        follow: bool = False,
        tail: Optional[int] = None,
    ) -> Generator[str, bool, None]:
        """Get the logs of a Seldon Core model deployment.

        Args:
            follow: if True, the logs will be streamed as they are written
            tail: only retrieve the last NUM lines of log output.

        Returns:
            A generator that can be accessed to get the service logs.
        """
        return self._get_client().get_deployment_logs(
            self.seldon_deployment_name,
            follow=follow,
            tail=tail,
        )

    @property
    def prediction_url(self) -> Optional[str]:
        """The prediction URI exposed by the prediction service.

        Returns:
            The prediction URI exposed by the prediction service, or None if
            the service is not yet ready.
        """
        from zenml.integrations.seldon.model_deployers.seldon_model_deployer import (
            SeldonModelDeployer,
        )

        if not self.is_running:
            return None
        namespace = self._get_client().namespace
        model_deployer = cast(
            SeldonModelDeployer,
            SeldonModelDeployer.get_active_model_deployer(),
        )
        return os.path.join(
            model_deployer.config.base_url,
            "seldon",
            namespace,
            self.seldon_deployment_name,
            "api/v0.1/predictions",
        )

    def predict(self, request: str) -> Any:
        """Make a prediction using the service.

        Args:
            request: a numpy array representing the request

        Returns:
            A numpy array representing the prediction returned by the service.

        Raises:
            Exception: if the service is not yet ready.
            ValueError: if the prediction_url is not set.
        """
        if not self.is_running:
            raise Exception(
                "Seldon prediction service is not running. "
                "Please start the service before making predictions."
            )

        if self.prediction_url is None:
            raise ValueError("`self.prediction_url` is not set, cannot post.")

        if isinstance(request, str):
            request = json.loads(request)
        else:
            raise ValueError("Request must be a json string.")
        response = requests.post(  # nosec
            self.prediction_url,
            json={"data": {"ndarray": request}},
        )
        response.raise_for_status()
        return response.json()
