#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Implementation of the vllm model deployer pipeline step."""

from typing import Optional, cast

from zenml import get_step_context, step
from zenml.integrations.vllm.model_deployers.vllm_model_deployer import (
    VLLMModelDeployer,
)
from zenml.integrations.vllm.services.vllm_deployment import (
    VLLMDeploymentService,
    VLLMServiceConfig,
)
from zenml.logger import get_logger

from constants import (
    DATASET_NAME_ARGILLA,
    DATASET_NAME_DISTILABEL,
    EMBEDDINGS_MODEL_ID_BASELINE,
    EMBEDDINGS_MODEL_ID_FINE_TUNED,
    EMBEDDINGS_MODEL_MATRYOSHKA_DIMS,
    SECRET_NAME,
)

logger = get_logger(__name__)


@step(enable_cache=False)
def vllm_model_deployer_step(
    port: int = 8000,
    tokenizer: Optional[str] = None,
    timeout: int = 1200,
    deploy_decision: bool = True,
) -> VLLMDeploymentService:
    """Model deployer pipeline step for vLLM.

    This step deploys a given Bento to a local vLLM http prediction server.

    Args:
        model: Name or path to huggingface model
        port: Port used by vllm server
        tokenizer: Name or path of the huggingface tokenizer to use.
            If unspecified, model name or path will be used.
        timeout: the number of seconds to wait for the service to start/stop.
        deploy_decision: whether to deploy the model or not

    Returns:
        vLLM deployment service
    """
    # get the current active model deployer
    model_deployer = cast(
        VLLMModelDeployer, VLLMModelDeployer.get_active_model_deployer()
    )

    # get pipeline name, step name and run id
    step_context = get_step_context()
    pipeline_name = step_context.pipeline.name
    step_name = step_context.step_run.name

    # create a config for the new model service
    predictor_cfg = VLLMServiceConfig(
        pipeline_name= pipeline_name,
        step_name=step_name,
        model_name=step_context.model.name,
        model_version=step_context.model.version,
        model=f"zenml/{EMBEDDINGS_MODEL_ID_FINE_TUNED}",
        served_model_name=step_context.model.name,
        port=port,
        tokenizer=tokenizer,
    )

    # create a new model deployment and replace an old one if it exists
    svc = model_deployer.deploy_model(
            replace=True,
            config=predictor_cfg,
            timeout=timeout,
            service_type=VLLMDeploymentService.SERVICE_TYPE,
        ),
    new_service = cast(
            VLLMDeploymentService,
            svc
        )

    logger.info(
        f"VLLM deployment service started and reachable at:\n"
        f"    {new_service.prediction_url}\n"
    )

    return new_service