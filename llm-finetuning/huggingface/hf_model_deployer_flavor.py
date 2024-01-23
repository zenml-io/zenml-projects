"""Huggingface model deployer flavor."""
from typing import Optional, Type, TYPE_CHECKING
from zenml.model_deployers.base_model_deployer import (
    BaseModelDeployerFlavor,
    BaseModelDeployerConfig,
)
from zenml.config.base_settings import BaseSettings
from huggingface.hf_deployment_base_config import HuggingFaceBaseConfig
from zenml.utils.secret_utils import SecretField

if TYPE_CHECKING:
    from huggingface.hf_model_deployer import HuggingFaceModelDeployer


HUGGINGFACE_MODEL_DEPLOYER_FLAVOR = "hfendpoint"


class HuggingFaceModelDeployerSettings(HuggingFaceBaseConfig, BaseSettings):
    """Settings for the Huggingface model deployer."""


class HuggingFaceModelDeployerConfig(
    BaseModelDeployerConfig, HuggingFaceModelDeployerSettings
):
    """Configuration for the Huggingface model deployer."""


class HuggingFaceModelDeployerFlavor(BaseModelDeployerFlavor):
    """Huggingface Endpoint model deployer flavor."""

    @property
    def name(self) -> str:
        """Name of the flavor.

        Returns:
            The name of the flavor.
        """
        return HUGGINGFACE_MODEL_DEPLOYER_FLAVOR

    @property
    def docs_url(self) -> Optional[str]:
        """A url to point at docs explaining this flavor.

        Returns:
            A flavor docs url.
        """
        return self.generate_default_docs_url()

    @property
    def sdk_docs_url(self) -> Optional[str]:
        """A url to point at SDK docs explaining this flavor.

        Returns:
            A flavor SDK docs url.
        """
        return self.generate_default_sdk_docs_url()

    @property
    def logo_url(self) -> str:
        """A url to represent the flavor in the dashboard.

        Returns:
            The flavor logo.
        """
        return "https://public-flavor-logos.s3.eu-central-1.amazonaws.com/model_registry/huggingface.png"

    @property
    def config_class(self) -> Type[HuggingFaceModelDeployerConfig]:
        """Returns `HuggingFaceModelDeployerConfig` config class.

        Returns:
            The config class.
        """
        return HuggingFaceModelDeployerConfig

    @property
    def implementation_class(self) -> Type["HuggingFaceModelDeployer"]:
        """Implementation class for this flavor.

        Returns:
            The implementation class.
        """
        from huggingface.hf_model_deployer import HuggingFaceModelDeployer

        return HuggingFaceModelDeployer
