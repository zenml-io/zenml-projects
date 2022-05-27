from typing import cast

import click
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
from rich import print
from steps.data_process import drop_cols, encode_cat_cols
from steps.data_splitter import data_splitter
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data
from steps.trainer import model_trainer
from zenml.integrations.seldon.model_deployers import SeldonModelDeployer
from zenml.integrations.seldon.services import (
    SeldonDeploymentConfig,
    SeldonDeploymentService,
)
from zenml.integrations.seldon.steps import (
    SeldonDeployerStepConfig,
    seldon_model_deployer_step,
)


@click.command()
@click.option(
    "--deploy",
    "-d",
    is_flag=True,
    help="Run the deployment pipeline to train and deploy a model",
)
@click.option(
    "--predict",
    "-p",
    is_flag=True,
    help="Run the inference pipeline to send a prediction request " "to the deployed model",
)
@click.option(
    "--min-accuracy",
    default=0.50,
    help="Minimum accuracy required to deploy the model (default: 0.50)",
)
def main(
    deploy: bool,
    predict: bool,
    min_accuracy: float,
):
    """Run the Seldon example continuous deployment or inference pipeline
    Example usage:
        python run.py --deploy --predict --min-accuracy 0.50
    """
    model_name = "model"
    deployment_pipeline_name = "continuous_deployment_pipeline"
    deployer_step_name = "seldon_model_deployer_step"

    model_deployer = SeldonModelDeployer.get_active_model_deployer()

    seldon_implementation = "SKLEARN_SERVER"

    if deploy:
        # Initialize a continuous deployment pipeline run
        deployment = continuous_deployment_pipeline(
            ingest_data=ingest_data(),
            encode_cat_cols=encode_cat_cols(),
            drop_cols=drop_cols(),
            data_splitter=data_splitter(),
            model_trainer=model_trainer(),
            evaluator=evaluation(),
            deployment_trigger=deployment_trigger(
                config=DeploymentTriggerConfig(
                    min_accuracy=min_accuracy,
                )
            ),
            model_deployer=seldon_model_deployer_step(
                config=SeldonDeployerStepConfig(
                    service_config=SeldonDeploymentConfig(
                        model_name=model_name,
                        replicas=1,
                        implementation=seldon_implementation,
                    ),
                    timeout=120,
                )
            ),
        )

        deployment.run()

    if predict:
        # Initialize an inference pipeline run
        inference = inference_pipeline(
            dynamic_importer=dynamic_importer(),
            prediction_service_loader=prediction_service_loader(
                SeldonDeploymentLoaderStepConfig(
                    pipeline_name=deployment_pipeline_name,
                    step_name=deployer_step_name,
                    model_name=model_name,
                )
            ),
            predictor=predictor(),
        )
        inference.run()

    services = model_deployer.find_model_server(
        pipeline_name=deployment_pipeline_name,
        pipeline_step_name=deployer_step_name,
        model_name=model_name,
    )
    if services:
        service = cast(SeldonDeploymentService, services[0])
        if service.is_running:
            print(
                f"The Seldon prediction server is running remotely as a Kubernetes "
                f"service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml served-models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The Seldon prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )

    else:
        print(
            "No Seldon prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
        )


if __name__ == "__main__":
    main()
