#  Copyright (c) ZenML GmbH 2022. All Rights Reserved.
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

from typing import Optional

from bentoml._internal.bento import bento
from zenml import get_step_context, step
from zenml.client import Client
from zenml.integrations.bentoml.services.bentoml_local_deployment import (
    BentoMLLocalDeploymentConfig,
    BentoMLLocalDeploymentService,
)
from zenml.logger import get_logger
from zenml.utils import source_utils

logger = get_logger(__name__)


@step
def deployment_deploy(
    bento: bento.Bento,
    target_env: str,
) -> Optional[BentoMLLocalDeploymentService]:
    # Deploy a model using the MLflow Model Deployer
    zenml_client = Client()
    step_context = get_step_context()
    pipeline_name = step_context.pipeline.name
    step_name = step_context.step_run.name
    model_deployer = zenml_client.active_stack.model_deployer
    bentoml_deployment_config = BentoMLLocalDeploymentConfig(
        model_name=step_context.model.name,
        model_version=target_env,
        description="An example of deploying a model using the MLflow Model Deployer",
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_uri=bento.info.labels.get("model_uri"),
        bento_tag=str(bento.tag),
        bento_uri=bento.info.labels.get("bento_uri"),
        working_dir=source_utils.get_source_root(),
        timeout=1500,
    )
    service = model_deployer.deploy_model(
        config=bentoml_deployment_config,
        service_type=BentoMLLocalDeploymentService.SERVICE_TYPE,
    )
    logger.info(
        f"The deployed service info: {model_deployer.get_model_server_info(service)}"
    )
    return service
