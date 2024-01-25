import click

from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)


@click.command()
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
@click.option(
    "--model_type",
    "-m",
    type=click.Choice(["lightgbm", "randomforest", "xgboost"]),
    default="xgboost",
    help="Here you can choose what type of model should be trained."
)
def run_main(stop_service: bool, model_type: str, model_name="Customer_Satisfaction_Predictor"):
    """Run the mlflow example pipeline"""
    if stop_service:
        # get the MLflow model deployer stack component
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        # fetch existing services with same pipeline name, step name and model name
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="model_deployer",
            model_name=model_name,
            running=True,
        )

        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    continuous_deployment_pipeline.with_options(config_path="config.yaml")(model_type=model_type)

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    inference_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri {get_tracking_uri()}\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step"
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon "
            f"process and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
            f"To stop the service, re-run the same command and supply the "
            f"`--stop-service` argument."
        )


if __name__ == "__main__":
    run_main()
