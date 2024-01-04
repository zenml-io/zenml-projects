# {% include 'templates/license_header' %}

import os
from typing import Optional

import click
from pipelines import (
    feature_engineering,
    inference,
    breast_cancer_training,
    breast_cancer_deployment_pipeline
)
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Starter project CLI v0.0.1.

Run the ZenML starter project with basic options.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the training pipeline
    python run.py --training-pipeline

  \b 
  # Run the training pipeline with versioned artifacts
    python run.py --training-pipeline --train-dataset-version-name=1 --test-dataset-version-name=1

  \b
  # Run the inference pipeline
    python run.py --inference-pipeline

"""
)
@click.option(
    "--train-dataset-name",
    default="dataset_trn",
    type=click.STRING,
    help="The name of the train dataset produced by feature engineering.",
)
@click.option(
    "--train-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the train dataset produced by feature engineering. "
    "If not specified, a new version will be created.",
)
@click.option(
    "--test-dataset-name",
    default="dataset_tst",
    type=click.STRING,
    help="The name of the test dataset produced by feature engineering.",
)
@click.option(
    "--test-dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the test dataset produced by feature engineering. "
    "If not specified, a new version will be created.",
)
@click.option(
    "--config",
    default=None,
    type=click.STRING,
    help="The name of the config",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)
@click.option(
    "--deployment-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that deploys the model.",
)
def main(
    train_dataset_name: str = "dataset_trn",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "dataset_tst",
    test_dataset_version_name: Optional[str] = None,
    config: Optional[str] = None,
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    deployment_pipeline: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline
    """
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        pipeline_args = {}
        pipeline_args["config_path"] = os.path.join(
            config_folder, "feature_engineering.yaml"
        )
        run_args_feature = {}
        feature_engineering.with_options(**pipeline_args)(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!")

    # Execute Training Pipeline
    if training_pipeline:
        pipeline_args = {}
        if config is None:
            pipeline_args["config_path"] = os.path.join(config_folder, "training.yaml")
        else:
            pipeline_args["config_path"] = os.path.join(config_folder, config)
        run_args_train = {}

        # If train_dataset_version_name is specified, use versioned artifacts
        if train_dataset_version_name or test_dataset_version_name:
            # However, both train and test dataset versions must be specified
            assert (
                train_dataset_version_name is not None
                and test_dataset_version_name is not None
            )
            client = Client()
            train_dataset_artifact = client.get_artifact(
                train_dataset_name, train_dataset_version_name
            )
            # If train dataset is specified, test dataset must be specified
            test_dataset_artifact = client.get_artifact(
                test_dataset_name, test_dataset_version_name
            )
            # Use versioned artifacts
            run_args_train["train_dataset_id"] = train_dataset_artifact.id
            run_args_train["test_dataset_id"] = test_dataset_artifact.id

        breast_cancer_training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline finished successfully!")

    if inference_pipeline:
        pipeline_args = {}
        if config is None:
            pipeline_args["config_path"] = os.path.join(config_folder, "inference.yaml")
        else:
            pipeline_args["config_path"] = os.path.join(config_folder, config) 
        run_args_inference = {}
        inference.with_options(**pipeline_args)(**run_args_inference)
        logger.info("Inference pipeline finished successfully!")

    if deployment_pipeline:
        pipeline_args = {}
        pipeline_args["config_path"] = os.path.join(config_folder, "deployment.yaml")
        run_args_inference = {}
        breast_cancer_deployment_pipeline.with_options(**pipeline_args)(**run_args_inference)
        logger.info("Deployment pipeline finished successfully!")

if __name__ == "__main__":
    main()
