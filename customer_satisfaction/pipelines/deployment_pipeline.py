import os 
import mlflow 
import json 
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
from io import StringIO
import json

from model.data_ingestion import IngestData 

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


model_deployer = mlflow_deployer_step(name="model_deployer")

class MLFlowDeploymentLoaderStepConfig(BaseStepConfig):
    """MLflow deployment getter configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
    """

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def prediction_service_loader(
    config: MLFlowDeploymentLoaderStepConfig, context: StepContext
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    service = load_last_service_from_step(
        pipeline_name=config.pipeline_name,
        step_name=config.step_name,
        step_context=context,
        running=config.running,
    )
    if not service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{config.step_name} step in the {config.pipeline_name} pipeline "
            f"is currently running."
        )

    return service

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started 
    data = json.loads(data)  
    data.pop("columns")   
    data.pop("index")
    columns_for_df = ["payment_sequential","payment_installments","payment_value","price","freight_value","product_name_lenght","product_description_lenght","product_photos_qty","product_weight_g","product_length_cm","product_height_cm","product_width_cm"]
    df = pd.DataFrame(data["data"],  columns=columns_for_df) 
    json_list = json.loads(json.dumps(list(df.T.to_dict().values()))) 
    data = np.array(json_list)
    prediction = service.predict(data) 
    return prediction  

@pipeline(enable_cache=True, requirements_file=requirements_file)
def continuous_deployment_pipeline(
    ingest_data, 
    clean_data, 
    model_train, 
    evaluation, 
    deployment_trigger,
    model_deployer,
): 
    # Link all the steps artifacts together
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model= model_train(
        x_train, x_test, y_train, y_test
    )
    r2_score, rmse = evaluation(model, x_test, y_test) 
    deployment_decision = deployment_trigger(accuracy=r2_score)
    model_deployer(deployment_decision)


@pipeline(enable_cache=True, requirements_file=requirements_file)
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, batch_data)




@step(enable_cache=False)
def dynamic_importer() -> Output(data=str):
    """Downloads the latest data from a mock API.""" 
    ingest_data = IngestData()
    data = ingest_data.get_data_for_test()
    return data


