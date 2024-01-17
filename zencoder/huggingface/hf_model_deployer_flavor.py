from zenml.model_deployers.base_model_deployer import (
    BaseModelDeployerFlavor,
    BaseModelDeployerConfig,
)


class HFConfig(BaseModelDeployerConfig):
    pass


class HFModelDeployerFlavor(BaseModelDeployerFlavor):
    """Huggingface Endpoint model deployer flavor."""

    @property
    def name(self) -> str:
        """Name of the flavor.

        Returns:
            The name of the flavor.
        """
        return "hfendpoint"

    @property
    def config_class(self):
        """Returns `HFConfig` config class.

        Returns:
                The config class.
        """
        return HFConfig

    @property
    def implementation_class(self):
        """Implementation class for this flavor.

        Returns:
            The implementation class.
        """
        from huggingface.hf_model_deployer import HFEndpointModelDeployer

        return HFEndpointModelDeployer
