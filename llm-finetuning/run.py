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
    "--config",
    type=str,
    default=None,
    help="Path to the YAML config file.",
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
    "--deploy-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that deploys the model.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--deployment-target",
    type=str,
    default="vllm",
    help="The target for the deployment pipeline.",
)
def main(
    config: str = None,
    deployment_target: str = "huggingface",
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    deploy_pipeline: bool = False,
    no_cache: bool = False,
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
    """
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {"enable_cache": not no_cache}
    if config:
        pipeline_args["config_path"] = os.path.join(config_folder, config)

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        from pipelines import generate_code_dataset
        generate_code_dataset.with_options(**pipeline_args)()
        logger.info("Feature Engineering pipeline finished successfully!\n")

    elif training_pipeline:
        from pipelines import finetune_starcoder

        finetune_starcoder.with_options(**pipeline_args)()
        logger.info("Training pipeline finished successfully!\n")

    elif deploy_pipeline:
        from pipelines import deployment_pipeline

        deployment_pipeline.with_options(**pipeline_args)(target=deployment_target)
        logger.info("Deployment pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
