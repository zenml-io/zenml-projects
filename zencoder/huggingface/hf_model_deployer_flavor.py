from zenml.model_deployers.base_model_deployer import (
    BaseModelDeployerConfig,
    BaseModelDeployerFlavor,
)


class HFInferenceEndpointConfig(BaseModelDeployerConfig):
    """Base class for all ZenML model deployer configurations."""

    endpoint_name: str
    revision: str
    repository: str
    framework: str
    task: str
    accelerator: str
    vendor: str
    region: str
    type: str
    instance_size: str
    instance_type: str
    hf_token: str


class HFModelDeployerFlavor(BaseModelDeployerFlavor):
    """Seldon Core model deployer flavor."""

    @property
    def name(self) -> str:
        """Name of the flavor.

        Returns:
            The name of the flavor.
        """
        return "hfendpoint"

    @property
    def config_class(self):
        """Returns `SeldonModelDeployerConfig` config class.

        Returns:
                The config class.
        """
        return HFInferenceEndpointConfig

    @property
    def implementation_class(self):
        """Implementation class for this flavor.

        Returns:
            The implementation class.
        """
        from huggingface.hf_model_deployer import HFEndpointModelDeployer

        return HFEndpointModelDeployer
