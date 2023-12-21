import os

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from zenml import pipeline

from pipelines.training_pipeline import customer_satisfaction_training_pipeline
from pipelines.utils import model_version
from steps import ingest_data, clean_data, train_model, evaluation, predictor
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader
from steps.prediction_service_loader import prediction_service_loader

requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline(
    enable_cache=False,
    model_version=model_version
)
def continuous_deployment_pipeline(
    model_type: str = "lightgbm"
):
    """Run a training job and deploy an mlflow model deployment."""
    # Run a training pipeline
    model, is_promoted = customer_satisfaction_training_pipeline(model_type=model_type)

    # Fetch the production model from the Model Registry
    production_model = model_loader(model_name=model_version.name, after="model_promoter")

    # Only actually redeploy the model if the most recent training
    #  led to a model promotion
    mlflow_model_deployer_step(
        workers=3,
        deploy_decision=is_promoted,
        model=production_model
    )


@pipeline(enable_cache=False)
def inference_pipeline():
    """Run a batch inference job with data loaded from an API."""
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
        running=True
    )
    predictor(model_deployment_service, batch_data)
