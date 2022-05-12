import click
from materializer.custom_materializer import cs_materializer
from pipelines.deployment_pipeline import (
    DeploymentTriggerConfig,
    MLFlowDeploymentLoaderStepConfig,
    continuous_deployment_pipeline,
    deployment_trigger,
    dynamic_importer,
    inference_pipeline,
    model_deployer,
    prediction_service_loader,
    predictor,
)
from rich import print
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.steps import MLFlowDeployerConfig
from zenml.services import load_last_service_from_step


@click.command()
@click.option(
    "--min-accuracy",
    default=1.8,
    help="Minimum mse required to deploy the model",
)
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(min_accuracy: float, stop_service: bool):

    """Run the mlflow example pipeline"""
    if stop_service:
        service = load_last_service_from_step(
            pipeline_name="continuous_deployment_pipeline",
            step_name="model_deployer",
            running=True,
        )
        if service:
            service.stop(timeout=10)
        return

    deployment = continuous_deployment_pipeline(
        ingest_data(),
        clean_data().with_return_materializers(cs_materializer),
        train_model(),
        evaluation(),
        deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_accuracy=min_accuracy,
            )
        ),
        model_deployer=model_deployer(config=MLFlowDeployerConfig(workers=3)),
    )
    deployment.run()

    inference = inference_pipeline(
        dynamic_importer=dynamic_importer().with_return_materializers(cs_materializer),
        prediction_service_loader=prediction_service_loader(
            MLFlowDeploymentLoaderStepConfig(
                pipeline_name="continuous_deployment_pipeline",
                step_name="model_deployer",
            )
        ),
        predictor=predictor(),
    )
    inference.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    service = load_last_service_from_step(
        pipeline_name="continuous_deployment_pipeline",
        step_name="model_deployer",
        running=True,
    )
    if service:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service.prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


# if __name__ == "__main__":
#     main()
