import json

import numpy as np
import pandas as pd
from materializer.customer_materializer import cs_materializer
from pipelines.seldon_deployment_pipeline import (
    DeploymentTriggerConfig,
    SeldonDeploymentLoaderStepConfig,
    continuous_deployment_pipeline,
    deployment_trigger,
    dynamic_importer,
    inference_pipeline,
    prediction_service_loader,
    predictor,
)
from sklearn.base import ClassifierMixin
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import (
    SeldonDeployerStepConfig,
    seldon_model_deployer_step,
)
from zenml.pipelines import pipeline
from zenml.repository import Repository
from zenml.steps import Output, StepContext, step

# @step
# def prediction(context: StepContext) -> ClassifierMixin:
#     pipeline_runs = context.metadata_store.get_pipeline("training_pipeline").runs
#     for run in pipeline_runs:
#         trainer_step = run.get_step("trainer")
#         model = trainer_step.output
#     return model


# @pipeline(requirements_file="kubeflow_requirements.txt")
# def prediction_pipeline(prediction):
#     prediction = prediction()


@step
def predictor(
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service"""

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
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    # prediction = service.predict(data)
    # prediction = prediction.argmax(axis=-1)
    # print("Prediction: ", prediction)
    # return prediction
    return data


if __name__ == "__main__":
    step = predictor().with_return_materializers(cs_materializer)
    print(step.get_materializers(ensure_complete=True))
