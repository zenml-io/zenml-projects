from .utils import get_data_for_test
import json
from typing import cast

import numpy as np  # type: ignore [import]
import pandas as pd
from zenml.integrations.constants import SELDON, SKLEARN, XGBOOST, LIGHTGBM
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import SeldonDeploymentService
from zenml.logger import get_logger
from zenml.pipelines import pipeline
from zenml.steps import BaseParameters, Output, step

logger = get_logger(__name__)


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float


@step(enable_cache=False)
def dynamic_importer() -> Output(data=pd.DataFrame):
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data


@step
def deployment_trigger(
    accuracy: float,
    params: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy > params.min_accuracy


class SeldonDeploymentLoaderStepConfig(BaseParameters):
    """Seldon deployment loader configuration
    Attributes:
        pipeline_name: name of the pipeline that deployed the Seldon prediction
            server
        step_name: the name of the step that deployed the Seldon prediction
            server
        model_name: the name of the model that was deployed
    """

    pipeline_name: str
    step_name: str
    model_name: str


@step(enable_cache=False)
def prediction_service_loader(
    params: SeldonDeploymentLoaderStepConfig,
) -> SeldonDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
        model_name=params.model_name,
    )
    if not services:
        raise RuntimeError(
            f"No Seldon Core prediction server deployed by the "
            f"'{params.step_name}' step in the '{params.pipeline_name}' "
            f"pipeline for the '{params.model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The Seldon Core prediction server last deployed by the "
            f"'{params.step_name}' step in the '{params.pipeline_name}' "
            f"pipeline for the '{params.model_name}' model is not currently "
            f"running."
        )

    return cast(SeldonDeploymentService, services[0])


@step
def predictor(
    service: SeldonDeploymentService,
    data: pd.DataFrame,
) -> Output(predictions=np.ndarray):
    """Run an inference request against a prediction service"""

    service.start(timeout=120)  # should be a NOP if already started
    data = data.to_json(orient="split")
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    data = np.array(df)
    predictions = service.predict(data)
    predictions = predictions.argmax(axis=-1)
    print("Prediction: ", predictions)
    return predictions


@pipeline
def continuous_deployment_pipeline(
    ingest_data,
    encode_cat_cols,
    drop_cols,
    data_splitter,
    model_trainer,
    evaluator,
    deployment_trigger,
    model_deployer,
):
    # Link all the steps and artifacts together
    customer_churn_df = ingest_data()
    customer_churn_df = encode_cat_cols(customer_churn_df)
    customer_churn_df = drop_cols(customer_churn_df)
    train, test = data_splitter(customer_churn_df)
    model = model_trainer(train)
    accuracy = evaluator(model, test)
    deployment_decision = deployment_trigger(accuracy=accuracy)
    model_deployer(deployment_decision, model)


@pipeline
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    # Link all the steps artifacts together
    inference_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, inference_data)
