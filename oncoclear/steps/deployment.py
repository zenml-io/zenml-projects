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

from typing import Annotated, Tuple
import os
import datetime
import docker

from zenml import step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils import docker_utils
from zenml import log_metadata

logger = get_logger(__name__)

# NOTE: This step is no longer used as we now take these values as direct inputs to the pipeline
# Keeping it here (commented out) for reference in case we want to reintroduce a secrets-based approach
"""
@step
def get_deployment_config(
    model_name: str,
    model_stage: str,
    secret_name: str = "deployment_service_key",
    secret_key: str = "key",
) -> Tuple[
    Annotated[str, "zenml_server_url"],
    Annotated[str, "zenml_api_key"],
    Annotated[str, "model_name"],
    Annotated[str, "model_stage"],
]:
    \"""Fetches deployment configuration: ZenML server URL and API key from secrets.

    Args:
        model_name: Name of the model to deploy.
        model_stage: Stage of the model version to deploy.
        secret_name: Name of the ZenML Secret containing the API key.
        secret_key: Key within the ZenML Secret that holds the actual API key value.

    Returns:
        Tuple containing ZenML server URL, API key, model name, and model stage.

    Raises:
        RuntimeError: If the specified secret or key is not found.
        KeyError: If the secret exists but doesn't contain the expected key.
    \"""
    logger.info(
        f"Fetching deployment configuration for model '{model_name}:{model_stage}' "
        f"using secret '{secret_name}' (key: '{secret_key}')."
    )
    client = Client()
    try:
        server_url = client.zen_store.url
        logger.info(f"ZenML Server URL: {server_url}")

        api_key_secret = client.get_secret(secret_name)
        api_key = api_key_secret.secret_values[secret_key]
        logger.info(f"Successfully fetched API key from secret '{secret_name}'.")

        # Basic validation to ensure API key looks somewhat reasonable (not empty)
        if not api_key:
             raise ValueError(f"API key value found in secret '{secret_name}' with key '{secret_key}' is empty.")

    except KeyError as e:
        logger.error(
            f"Secret '{secret_name}' found, but it does not contain the key "
            f"'{secret_key}'. Please ensure the secret is created correctly "
            f"with the API key stored under the key '{secret_key}'."
        )
        # Re-raise as KeyError to indicate the specific issue
        raise KeyError(
            f"Secret '{secret_name}' does not contain key '{secret_key}'."
        ) from e
    except Exception as e:
        # Catch potential errors during secret fetching (e.g., secret not found)
        logger.error(
            f"Failed to fetch deployment secret '{secret_name}' or ZenML server URL. "
            f"Please ensure the secret '{secret_name}' exists and contains the key '{secret_key}' "
            f"with the deployment API key. Error: {e}"
        )
        # Wrap generic exceptions in a RuntimeError for clarity
        if isinstance(e, (KeyError, ValueError)): # Don't wrap our specific errors
             raise e
        raise RuntimeError(f"Failed to get deployment configuration: {e}") from e

    # Pass through model name and stage for the next steps
    return server_url, api_key, model_name, model_stage
"""


@step(
    enable_cache=False
)  # Avoid caching image builds unless inputs are identical
def build_deployment_image(
    model_name: str,
    model_stage: str,
) -> Annotated[str, "image_name"]:
    """Builds a Docker image for the FastAPI deployment service.

    Args:
        model_name: Name of the model being deployed (used for tagging).
        model_stage: Stage of the model being deployed (used for tagging).

    Returns:
        The name and tag of the built Docker image.
    """
    # Define image name based on model name and stage
    image_name = f"local-deployment-{model_name}:{model_stage}"
    logger.info(f"Building Docker image: {image_name}")

    # Define paths relative to the project root
    # Assumes this script is in 'steps/' and 'api/' is at the project root
    project_root = os.path.join(os.path.dirname(__file__), "..")
    build_context_path = os.path.abspath(os.path.join(project_root, "api"))
    dockerfile_path = os.path.abspath(
        os.path.join(build_context_path, "Dockerfile")
    )
    utils_path = os.path.abspath(os.path.join(project_root, "utils"))

    logger.info(f"Using build context: {build_context_path}")
    logger.info(f"Using Dockerfile: {dockerfile_path}")
    logger.info(f"Utils module path: {utils_path}")

    # Check if Dockerfile exists
    if not os.path.exists(dockerfile_path):
        raise FileNotFoundError(f"Dockerfile not found at: {dockerfile_path}")

    # Copy the utils directory to the api directory so it's available in the build context
    utils_in_context = os.path.join(build_context_path, "utils")

    # Create utils directory in the build context if it doesn't exist
    if not os.path.exists(utils_in_context):
        os.makedirs(utils_in_context, exist_ok=True)
        logger.info(
            f"Created utils directory in build context: {utils_in_context}"
        )

    # Copy utils module files
    import shutil

    for item in os.listdir(utils_path):
        src = os.path.join(utils_path, item)
        dst = os.path.join(utils_in_context, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} to {dst}")
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info(f"Copied directory {src} to {dst}")

    try:
        # Build the image using ZenML's utility
        docker_utils.build_image(
            image_name=image_name,
            dockerfile=dockerfile_path,
            build_context_root=build_context_path,
            # Add any custom build options if needed, e.g.:
            # custom_build_options={"platform": "linux/amd64"}
        )
        logger.info(f"Successfully built Docker image: {image_name}")
    except Exception as e:
        logger.error(
            f"Failed to build Docker image '{image_name}'. Error: {e}"
        )
        raise RuntimeError(f"Docker image build failed: {e}") from e

    return image_name


# --- Add run_deployment_container step below ---
@step(enable_cache=False)  # Avoid caching container runs
def run_deployment_container(
    zenml_server_url: str,
    zenml_api_key: str,
    model_name: str,
    model_stage: str,
    image_name: str,
    model_artifact_name: str = "sklearn_classifier",
    preprocess_pipeline_name: str = "preprocess_pipeline",
    host_port: int = 8000,
    container_port: int = 8000,
) -> Tuple[
    Annotated[str, "container_id"],
    Annotated[str, "service_url"],
]:
    """Runs the Docker container for the model deployment service and logs deployment metadata.

    Args:
        zenml_server_url: URL of the ZenML server.
        zenml_api_key: API key for the ZenML server.
        model_name: Name of the model to deploy.
        model_stage: Stage of the model to deploy.
        image_name: Name of the Docker image to run.
        model_artifact_name: Name of the model artifact to load (default: "sklearn_classifier").
        preprocess_pipeline_name: Name of the preprocessing pipeline artifact (default: "preprocess_pipeline").
        host_port: Port to expose on the host.
        container_port: Port the container is listening on.

    Returns:
        Tuple containing the container ID and service URL.
    """
    logger.info(f"Preparing to run container from image: {image_name}")

    # Create a Docker client
    client = docker.from_env()

    # Check if the image exists
    try:
        client.images.get(image_name)
        logger.info(f"Found Docker image: {image_name}")
    except docker.errors.ImageNotFound:
        raise RuntimeError(
            f"Docker image '{image_name}' not found. Please build it first."
        )

    # Define environment variables for the container
    env_vars = {
        "ZENML_STORE_URL": zenml_server_url,
        "ZENML_STORE_API_KEY": zenml_api_key,
        "MODEL_NAME": model_name,
        "MODEL_STAGE": model_stage,
        "MODEL_ARTIFACT_NAME": model_artifact_name,
        "PREPROCESS_PIPELINE_NAME": preprocess_pipeline_name,
        "PORT": str(container_port),
    }

    # Debug: Check the API key (mask it partially for logs)
    if zenml_api_key:
        masked_key = (
            zenml_api_key[:15] + "..." + zenml_api_key[-10:]
            if len(zenml_api_key) > 30
            else "***masked***"
        )
        logger.info(f"Using ZenML server: {zenml_server_url}")
        logger.info(f"Using API key (masked): {masked_key}")
    else:
        logger.warning("No API key provided! Authentication will likely fail.")

    # Define port mapping
    ports = {f"{container_port}/tcp": host_port}

    # Define a unique container name based on model name and stage
    container_name = (
        f"zenml-deployment-{model_name}-{model_stage}".lower().replace(
            "_", "-"
        )
    )

    # Check if a container with this name already exists and remove it if it does
    try:
        existing_container = client.containers.get(container_name)
        logger.warning(
            f"Found existing container '{container_name}'. Stopping and removing it."
        )
        existing_container.stop()
        existing_container.remove()
    except docker.errors.NotFound:
        # Container doesn't exist, which is fine
        pass

    try:
        # Run the container
        logger.info(
            f"Starting container '{container_name}' with image '{image_name}'"
        )
        container = client.containers.run(
            image=image_name,
            name=container_name,
            environment=env_vars,
            ports=ports,
            detach=True,  # Run in background
            restart_policy={"Name": "unless-stopped"},  # Restart if it crashes
        )

        # Ensure the env vars are passed correctly
        logger.info("Verifying environment variables in the container...")
        # Give the container a moment to start
        import time

        time.sleep(2)

        try:
            # Don't do this in production, this is just for debugging
            env_output = container.exec_run("env")
            if env_output.exit_code == 0:
                # Safely log env without showing full API key
                env_lines = env_output.output.decode("utf-8").split("\n")
                for line in env_lines:
                    if line.startswith("ZENML_STORE_API_KEY="):
                        key = line.split("=", 1)[1]
                        masked = (
                            key[:15] + "..." + key[-10:]
                            if len(key) > 30
                            else "***masked***"
                        )
                        logger.info(f"ZENML_STORE_API_KEY={masked}")
                    elif line.startswith("ZENML_"):
                        logger.info(line)
                    elif line.startswith("MODEL_"):
                        logger.info(line)
        except Exception as e:
            logger.warning(f"Could not verify environment variables: {e}")

        container_id = container.id
        service_url = f"http://localhost:{host_port}"

        logger.info(f"Container started successfully!")
        logger.info(f"Container ID: {container_id}")
        logger.info(f"Service URL: {service_url}")
        logger.info(f"API Documentation: {service_url}/docs")

        # Log deployment metadata directly here instead of in a separate step
        logger.info(
            f"Logging deployment metadata for model '{model_name}:{model_stage}'"
        )

        # Get updated container details
        container_info = client.containers.get(container_id).attrs

        # Create metadata to log
        current_time = datetime.datetime.now().isoformat()
        deployment_metadata = {
            "deployment_info": {
                "deployed_at": current_time,
                "deployed_by": os.environ.get("USER", "unknown"),
                "service_url": service_url,
                "api_docs_url": f"{service_url}/docs",
                "container_id": container_id,
                "container_name": container_info.get("Name", "").strip("/"),
                "container_image": container_info.get("Config", {}).get(
                    "Image", ""
                ),
                "container_status": container_info.get("State", {}).get(
                    "Status", ""
                ),
                "model_artifact_name": model_artifact_name,
            },
            "environment_info": {
                "host_platform": os.environ.get("OS", "unknown"),
                "zenml_version": os.environ.get("ZENML_VERSION", "unknown"),
                "deployed_from": os.environ.get("PWD", "unknown"),
            },
        }

        # Log the metadata
        log_metadata(
            metadata=deployment_metadata,
            model_name=model_name,
            model_version=model_stage,
        )

        logger.info("Successfully logged deployment metadata to model")

        return container_id, service_url

    except docker.errors.APIError as e:
        logger.error(f"Failed to run container: {e}")
        raise RuntimeError(f"Docker container creation failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error running container: {e}")
        raise RuntimeError(f"Failed to run deployment container: {e}") from e
