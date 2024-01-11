# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
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
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML NLP project CLI v0.0.1.

Deletes the endpoint of the latest run of the deployment pipeline.
"""
)
@click.option(
    "--deployment-pipeline-name",
    default="sentinment_analysis_deploy_pipeline",
    type=click.STRING,
    help="Name of the Deployment Pipeline.",
)
@click.option(
    "--deployment-pipeline-version",
    default=1,
    type=click.INT,
    help="Version of the Deployment Pipeline.",
)
@click.option(
    "--step-name",
    default="deploy_hf_to_sagemaker",
    type=click.STRING,
    help="Name of the step that returns the endpoint.",
)
@click.option(
    "--step-output-name",
    default="sagemaker_endpoint_name",
    type=click.STRING,
    help="Name of the step output that returns the endpoint.",
)
def main(
    deployment_pipeline_name: str = "sentinment_analysis_deploy_pipeline",
    deployment_pipeline_version: int = 1,
    step_name: str = "deploy_hf_to_sagemaker",
    step_output_name: str = "sagemaker_endpoint_name",
):
    """Main entry point for the script."""

    client = Client()
    latest_run = client.get_pipeline(
        deployment_pipeline_name, version=deployment_pipeline_version
    ).runs[0]
    endpoint_name = latest_run.steps[step_name].outputs[step_output_name].load()

    logger.info(f"Deleting endpoint with name: {endpoint_name}")
    # Do a `aws sagemaker delete-endpoint --endpoint-name <endpoint_name>` on the CLI
    # Throw an error if error code is not 0
    return_code = os.system(
        f"aws sagemaker delete-endpoint --endpoint-name {endpoint_name}"
    )
    if return_code != 0:
        raise RuntimeError("Endpoint could not be deleted!")
    logger.info("Endpoint deleted successfully!")


if __name__ == "__main__":
    main()
