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
from pipelines import (
    etl_pipeline,
    feature_engineering_pipeline,
    model_training_pipeline,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML ECB Interest Rate project.

Run the ZenML pipelines with basic options.

Examples:

  \b
  # Run the ETL pipeline
    python run.py --etl

  \b
  # Run the feature engineering pipeline
    python run.py --feature
  
  \b
  # Run the model training pipeline
    python run.py --training

  \b
  # Run all pipelines in sequence
    python run.py --etl --feature --training

"""
)
@click.option(
    "--etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
@click.option(
    "--feature",
    is_flag=True,
    default=False,
    help="Whether to run the feature engineering pipeline.",
)
@click.option(
    "--training",
    is_flag=True,
    default=False,
    help="Whether to run the model training pipeline.",
)
@click.option(
    "--mode",
    type=click.Choice(["develop", "production"]),
    default="develop",
    help="The mode in which to run the pipelines.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    etl: bool = False,
    feature: bool = False,
    training: bool = False,
    mode: str = "develop",
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution."""
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute ETL Pipeline
    if etl:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"etl_{mode}.yaml"
        )
        etl_pipeline.with_options(**pipeline_args)()
        logger.info("ETL pipeline finished successfully!\n")

    # Execute Feature Engineering Pipeline
    if feature:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"feature_engineering_{mode}.yaml"
        )
        feature_engineering_pipeline.with_options(**pipeline_args)()
        logger.info("Feature Engineering pipeline finished successfully!\n")

    # Execute Model Training Pipeline
    if training:
        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"training_{mode}.yaml"
        )
        model_training_pipeline.with_options(**pipeline_args)()
        logger.info("Model Training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
