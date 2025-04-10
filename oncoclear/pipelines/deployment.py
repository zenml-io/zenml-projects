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

from zenml import pipeline
from zenml.logger import get_logger
from zenml.client import Client

from steps.deployment import (
    build_deployment_image,
    run_deployment_container,
)

logger = get_logger(__name__)


@pipeline(enable_cache=False)  # Disable caching for the entire pipeline
def local_deployment(
    model_name: str,
    zenml_server_url: str = None,
    zenml_api_key: str = None,
    model_stage: str = "production",
    model_artifact_name: str = "sklearn_classifier",
    preprocess_pipeline_name: str = "preprocess_pipeline",
    host_port: int = 8000,
    container_port: int = 8000,
):
    """
    Model deployment pipeline.

    This pipeline builds a Docker image for a FastAPI service that serves the model,
    and runs the container locally.

    Args:
        model_name: Name of the ZenML model to deploy.
        zenml_server_url: URL of the ZenML server. If None, uses the current client's server URL.
        zenml_api_key: API key for the ZenML server. Required for the container to authenticate.
        model_stage: Stage of the model to deploy (default: "production").
        model_artifact_name: Name of the model artifact to load (default: "sklearn_classifier").
        preprocess_pipeline_name: Name of the preprocessing pipeline artifact (default: "preprocess_pipeline").
        host_port: Port to expose on the host (default: 8000).
        container_port: Port the container will listen on (default: 8000).
    """
    # If not provided, get the server URL from the current client
    if zenml_server_url is None:
        client = Client()
        zenml_server_url = client.zen_store.url
        logger.info(f"Using current ZenML server URL: {zenml_server_url}")

    # Validate API key is provided
    if zenml_api_key is None:
        logger.warning(
            "No API key provided. The deployment container will not be able to "
            "authenticate with the ZenML server unless environment variables "
            "are properly set."
        )

    # Build the deployment image
    image_name = build_deployment_image(
        model_name=model_name,
        model_stage=model_stage,
    )

    # Run the deployment container and log metadata
    container_id, service_url = run_deployment_container(
        zenml_server_url=zenml_server_url,
        zenml_api_key=zenml_api_key,
        model_name=model_name,
        model_stage=model_stage,
        image_name=image_name,
        model_artifact_name=model_artifact_name,
        preprocess_pipeline_name=preprocess_pipeline_name,
        host_port=host_port,
        container_port=container_port,
    )

    logger.info(f"Model '{model_name}:{model_stage}' deployed successfully!")
    logger.info(f"Service URL: {service_url}")
    logger.info(f"API Documentation: {service_url}/docs") 