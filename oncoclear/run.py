# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from typing import Optional

import click
import yaml
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import (
    feature_engineering,
    inference,
    training,
    local_deployment,
)

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Starter project.

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

  \b
  # Deploy a model locally with FastAPI
    python run.py --deploy-locally --deployment-model-name=my_model

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
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--deploy-locally",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that deploys a model locally with FastAPI.",
)
@click.option(
    "--deployment-model-name",
    default=None,
    type=click.STRING,
    help="Name of the model to deploy locally. Required if --deploy-locally is set.",
)
@click.option(
    "--deployment-model-stage",
    default="production",
    type=click.STRING,
    help="Stage of the model to deploy (default: 'production').",
)
@click.option(
    "--deployment-model-artifact-name",
    default="sklearn_classifier",
    type=click.STRING,
    help="Name of the model artifact to load (default: 'sklearn_classifier').",
)
@click.option(
    "--deployment-preprocess-pipeline-name",
    default="preprocess_pipeline",
    type=click.STRING,
    help="Name of the preprocessing pipeline artifact to load (default: 'preprocess_pipeline').",
)
@click.option(
    "--deployment-port",
    default=8000,
    type=click.INT,
    help="Port to expose the deployment server on (default: 8000).",
)
@click.option(
    "--deployment-zenml-server",
    default=None,
    type=click.STRING,
    help="URL of the ZenML server to use for deployment. If not provided, uses the current client's server.",
)
@click.option(
    "--deployment-zenml-api-key",
    default=None,
    type=click.STRING,
    help="API key for the ZenML server. Required for the container to authenticate if not set in environment.",
)
def main(
    train_dataset_name: str = "dataset_trn",
    train_dataset_version_name: Optional[str] = None,
    test_dataset_name: str = "dataset_tst",
    test_dataset_version_name: Optional[str] = None,
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    no_cache: bool = False,
    deploy_locally: bool = False,
    deployment_model_name: Optional[str] = None,
    deployment_model_stage: str = "production",
    deployment_model_artifact_name: str = "sklearn_classifier",
    deployment_preprocess_pipeline_name: str = "preprocess_pipeline",
    deployment_port: int = 8000,
    deployment_zenml_server: Optional[str] = None,
    deployment_zenml_api_key: Optional[str] = None,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        train_dataset_name: The name of the train dataset produced by feature engineering.
        train_dataset_version_name: Version of the train dataset produced by feature engineering.
            If not specified, a new version will be created.
        test_dataset_name: The name of the test dataset produced by feature engineering.
        test_dataset_version_name: Version of the test dataset produced by feature engineering.
            If not specified, a new version will be created.
        feature_pipeline: Whether to run the pipeline that creates the dataset.
        training_pipeline: Whether to run the pipeline that trains the model.
        inference_pipeline: Whether to run the pipeline that performs inference.
        no_cache: If `True` cache will be disabled.
        deploy_locally: Whether to run the pipeline that deploys a model locally with FastAPI.
        deployment_model_name: Name of the model to deploy locally.
        deployment_model_stage: Stage of the model to deploy.
        deployment_model_artifact_name: Name of the model artifact to load.
        deployment_preprocess_pipeline_name: Name of the preprocessing pipeline artifact to load.
        deployment_port: Port to expose the deployment server on.
        deployment_zenml_server: URL of the ZenML server for deployment.
        deployment_zenml_api_key: API key for the ZenML server.
    """
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "feature_engineering.yaml"
        )
        run_args_feature = {}
        feature_engineering.with_options(**pipeline_args)(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!\n")

        train_dataset_artifact = client.get_artifact_version(
            train_dataset_name
        )
        test_dataset_artifact = client.get_artifact_version(test_dataset_name)
        logger.info(
            "The latest feature engineering pipeline produced the following "
            f"artifacts: \n\n1. Train Dataset - Name: {train_dataset_name}, "
            f"Version Name: {train_dataset_artifact.version} \n2. Test Dataset: "
            f"Name: {test_dataset_name}, Version Name: {test_dataset_artifact.version}"
        )

    # Execute Training Pipeline
    if training_pipeline:
        run_args_train = {}

        # If train_dataset_version_name is specified, use versioned artifacts
        if train_dataset_version_name or test_dataset_version_name:
            # However, both train and test dataset versions must be specified
            assert (
                train_dataset_version_name is not None
                and test_dataset_version_name is not None
            )
            train_dataset_artifact_version = client.get_artifact_version(
                train_dataset_name, train_dataset_version_name
            )
            # If train dataset is specified, test dataset must be specified
            test_dataset_artifact_version = client.get_artifact_version(
                test_dataset_name, test_dataset_version_name
            )
            # Use versioned artifacts
            run_args_train["train_dataset_id"] = (
                train_dataset_artifact_version.id
            )
            run_args_train["test_dataset_id"] = (
                test_dataset_artifact_version.id
            )

        # Run the SGD pipeline
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "training_sgd.yaml"
        )
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with SGD finished successfully!\n\n")

        # Run the RF pipeline
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "training_rf.yaml"
        )
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with RF finished successfully!\n\n")

    if inference_pipeline:
        run_args_inference = {}
        pipeline_args = {"enable_cache": False}
        pipeline_args["config_path"] = os.path.join(
            config_folder, "inference.yaml"
        )

        # Configure the pipeline
        inference_configured = inference.with_options(**pipeline_args)

        # Fetch the production model
        with open(pipeline_args["config_path"], "r") as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        zenml_model = client.get_model_version(
            config["model"]["name"], config["model"]["version"]
        )
        preprocess_pipeline_artifact = zenml_model.get_artifact(
            "preprocess_pipeline"
        )

        # Use the metadata of feature engineering pipeline artifact
        #  to get the random state and target column
        random_state = preprocess_pipeline_artifact.run_metadata[
            "random_state"
        ]
        target = preprocess_pipeline_artifact.run_metadata["target"]
        run_args_inference["random_state"] = random_state
        run_args_inference["target"] = target

        # Run the pipeline
        inference_configured(**run_args_inference)
        logger.info("Inference pipeline finished successfully!")

    if deploy_locally:
        # Ensure model name is provided
        if not deployment_model_name:
            raise ValueError(
                "Model name must be provided for local deployment. "
                "Use --deployment-model-name to specify the model name."
            )

        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False

        # ZenML requires a config, but we don't need a specific one for deployment
        # So we'll just use a default config path, or later you can create a deployment.yaml
        # pipeline_args["config_path"] = os.path.join(config_folder, "deployment.yaml")

        run_args_deployment = {
            "model_name": deployment_model_name,
            "model_stage": deployment_model_stage,
            "model_artifact_name": deployment_model_artifact_name,
            "preprocess_pipeline_name": deployment_preprocess_pipeline_name,
            "host_port": deployment_port,
            "zenml_server_url": deployment_zenml_server,
            "zenml_api_key": deployment_zenml_api_key,
        }

        # Run the deployment pipeline
        local_deployment.with_options(**pipeline_args)(**run_args_deployment)

        logger.info(
            f"Local deployment pipeline for model '{deployment_model_name}:{deployment_model_stage}' "
            f"finished successfully!\n\n"
            f"The model is now accessible via FastAPI at http://localhost:{deployment_port}\n"
            f"API documentation is available at http://localhost:{deployment_port}/docs"
        )


if __name__ == "__main__":
    main()
