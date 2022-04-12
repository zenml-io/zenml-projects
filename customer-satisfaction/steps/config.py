from zenml.steps import BaseStepConfig


class ModelNameConfig(BaseStepConfig):
    """Model Configurations"""

    model_name: str = "lightgbm"
    fine_tuning: bool = False
