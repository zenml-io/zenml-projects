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
from zenml import Model
from pipelines import train_and_infer_statsmodel

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Time Series Example.

Using statsmodel to predict timeseries.

"""
)
@click.option(
    "--config",
    type=str,
    default="train_statsmodel.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--customer",
    type=str,
    default="acme",
    help="Name of the customer",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(
    config: str = "train_statsmodel.yaml",
    customer: str = "acme",
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution."""
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {"enable_cache": not no_cache}
    if config:
        pipeline_args["config_path"] = os.path.join(config_folder, config)

    logger.info(f"Configuring model with customer name: {customer}")
    pipeline_args["model"] = Model(
        name = f"sarixmax_{customer}_forecast",
    )
    
    if customer == "acme":
        data_stream = "CPIAUCSL"
    else:
        data_stream = "CPIAUCSL"

    train_and_infer_statsmodel.with_options(**pipeline_args)(data_stream=data_stream)
    logger.info("Training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
