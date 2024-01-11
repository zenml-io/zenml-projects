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
"""BentoML model deployer flavor."""

from typing import TYPE_CHECKING, Optional, Type

from zenml.model_deployers.base_model_deployer import (
    BaseModelDeployerConfig,
    BaseModelDeployerFlavor,
)
from zenml.models import ServiceConnectorRequirements

if TYPE_CHECKING:
    from hf_sagemaker_model_deployer import HFSagemakerModelDeployer


HF_SAGEMAKER_MODEL_DEPLOYER_FLAVOR = "hf_sagemaker"


class HFSagemakerModelDeployerConfig(  # type: ignore[misc] # https://github.com/pydantic/pydantic/issues/4173
    BaseModelDeployerConfig
):
    """HFSagemakerModelDeployerConfig orchestrator base config."""

    iam_role_arn: Optional[str] = None

    @property
    def is_local(self) -> bool:
        """Checks if this stack component is running locally.

        This designation is used to determine if the stack component can be
        shared with other users or if it is only usable on the local host.

        Returns:
            True if this config is for a local component, False otherwise.
        """
        return True


class HFSagemakerModelDeployerFlavor(BaseModelDeployerFlavor):
    """Flavor for the Huggingface Sagemaker model deployer."""

    @property
    def name(self) -> str:
        """Name of the flavor.

        Returns:
            Name of the flavor.
        """
        return HF_SAGEMAKER_MODEL_DEPLOYER_FLAVOR

    @property
    def config_class(self) -> Type[HFSagemakerModelDeployerConfig]:
        """Returns `HFSagemakerModelDeployerConfig` config class.

        Returns:
                The config class.
        """
        return HFSagemakerModelDeployerConfig

    @property
    def implementation_class(self) -> Type["HFSagemakerModelDeployer"]:
        """Implementation class for this flavor.

        Returns:
            The implementation class.
        """
        from hf_sagemaker_model_deployer import (
            HFSagemakerModelDeployer,
        )

        return HFSagemakerModelDeployer

    @property
    def service_connector_requirements(
        self,
    ) -> Optional[ServiceConnectorRequirements]:
        """Service connector resource requirements for service connectors.

        Specifies resource requirements that are used to filter the available
        service connector types that are compatible with this flavor.

        Returns:
            Requirements for compatible service connectors, if a service
            connector is required for this flavor.
        """
        return ServiceConnectorRequirements(
            resource_type="aws-generic",
        )
