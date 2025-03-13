#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os

import click
from pipelines.agent_creator import zenml_agent_creation_pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML Starter project.

Run the ZenML starter project with basic options.

Examples:

  \b
  # Run the pipeline with config.yaml in the configs folder
    python run.py --config config.yaml

"""
)
@click.option(
    "--config",
    type=str,
    default="agent_config.yaml",
    help="Path to the YAML config file.",
)
def main(
    config: str = "agent_config.yaml",
):
    """Main entry point for the pipeline execution."""
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {}
    if config:
        pipeline_args["config_path"] = os.path.join(config_folder, config)

    zenml_agent_creation_pipeline.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
