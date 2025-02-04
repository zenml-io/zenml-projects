# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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

import click
import yaml
from pipelines import (
    inference,
    training,
)

from zenml.client import Client
from zenml.logger import get_logger

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

"""
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
def main(
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        training_pipeline: Whether to run the pipeline that trains the model.
        inference_pipeline: Whether to run the pipeline that performs inference.
        no_cache: If `True` cache will be disabled.
    """
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute Training Pipeline
    if training_pipeline:
        run_args_train = {}

        # Run the SGD pipeline
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, "training_sgd.yaml"
        )
        training.with_options(**pipeline_args)(**run_args_train)
        logger.info("Training pipeline with SGD finished successfully!\n\n")

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
        ].value
        target = preprocess_pipeline_artifact.run_metadata["target"].value
        run_args_inference["random_state"] = random_state
        run_args_inference["target"] = target

        # Run the pipeline
        inference_configured(**run_args_inference)
        logger.info("Inference pipeline finished successfully!")


if __name__ == "__main__":
    main()
