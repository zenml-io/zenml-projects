import mlflow
import numpy as np  
import pandas as pd  
import requests  
import tensorflow as tf  

from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_deployer_step
from zenml.pipelines import pipeline
from zenml.services import load_last_service_from_step
from zenml.steps import BaseStepConfig, Output, StepContext, step

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")

class DeploymentTriggerConfig(BaseStepConfig):
    """Parameters that are used to trigger the deployment"""
    min_accuracy: float

@step
def deployment_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > config.min_accuracy
