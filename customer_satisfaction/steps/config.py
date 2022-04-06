from zenml.steps import BaseStepConfig

class ModelNameConfig(BaseStepConfig):
    model_name: str = "lightgbm"  
    fine_tuning: bool = False 
