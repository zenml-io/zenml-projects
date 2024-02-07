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
ZenML Credit and Risk Assessment project using LLMs.

Run this project with basic options.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature-pipeline
  
  \b
  # Run the training pipeline
    python run.py --training-pipeline

  \b
  # Run the deployment pipeline
    python run.py --deployment-pipeline

"""
)
@click.option(
    "--config_fe",
    type=str,
    default="default_feature_engineering.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--config_train",
    type=str,
    default="default_train.yaml",
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
    "--deployment-pipeline",
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
def main(
    config_fe: str = None,
    config_train: str = None,
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    deployment_pipeline: bool = False,
    no_cache: bool = False,
    **kwargs
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments, but most
        of which comes from the YAML config files)
      * launching the pipeline

    Args:
        config: Path to the YAML config file.
        feature_pipeline: Whether to run the pipeline that creates the dataset.
        training_pipeline: Whether to run the pipeline that trains the model.
        deployment_pipeline: Whether to run the pipeline that performs deployment.
        no_cache: If `True` cache will be disabled.
        kwargs: Additional arguments used for training classes.
    """
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {"enable_cache": not no_cache}

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        from pipelines import feature_engineering_pipeline

        if config_fe:
            pipeline_args["config_path"] = os.path.join(config_folder, config_fe)

        feature_engineering_pipeline.with_options(**pipeline_args)()
        logger.info("Feature Engineering pipeline finished successfully!\n")

    elif training_pipeline:
        from pipelines import training_pipeline

        if config_train:
            pipeline_args["config_path"] = os.path.join(config_folder, config_train)
        training_pipeline.with_options(**pipeline_args)()
        logger.info("Training pipeline finished successfully!\n")

    # elif deploy_pipeline:
    #     from pipelines import huggingface_deployment

    #     huggingface_deployment.with_options(**pipeline_args)()
    #     logger.info("Deployment pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
