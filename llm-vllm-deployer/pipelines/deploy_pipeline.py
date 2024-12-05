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

from zenml import pipeline
from typing import Annotated
from steps.vllm_deployer import vllm_model_deployer_step
from zenml.integrations.vllm.services.vllm_deployment import VLLMDeploymentService


@pipeline()
def deploy_vllm_pipeline(
    model: str,
    timeout: int = 1200,
) -> Annotated[VLLMDeploymentService, "GPT2"]:
    service = vllm_model_deployer_step(
        model=model,
        timeout=timeout,
    )
    return service
