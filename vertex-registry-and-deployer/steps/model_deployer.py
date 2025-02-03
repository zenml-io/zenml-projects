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

from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.gcp.services.vertex_deployment import (
    VertexAIDeploymentConfig,
    VertexDeploymentService,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def model_deployer(
    model_registry_uri: str,
) -> Annotated[
    VertexDeploymentService, ArtifactConfig(name="vertex_deployment", is_deployment_artifact=True)
]:
    """Model deployer step.
    
    Args:
        model_registry_uri: URI of the model in the model registry.
    
    Returns:
        The deployed model service.
    """
    zenml_client = Client()
    current_model = get_step_context().model
    model_deployer = zenml_client.active_stack.model_deployer
    vertex_deployment_config = VertexAIDeploymentConfig(
        location="europe-west1",
        name="zenml-vertex-quickstart",
        model_name=current_model.name,
        description="An example of deploying a model using the MLflow Model Deployer",
        model_id=model_registry_uri,
    )
    service = model_deployer.deploy_model(
        config=vertex_deployment_config,
        service_type=VertexDeploymentService.SERVICE_TYPE,
    )

    logger.info(
        f"The deployed service info: {model_deployer.get_model_server_info(service)}"
    )
    return service
