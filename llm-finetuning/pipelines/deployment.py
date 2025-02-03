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

from steps import deploy_model_to_hf_hub, vllm_model_deployer_step
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def deployment_pipeline(target: str = "huggingface"):
    """This pipeline pushes the model to the hub."""
    if target == "huggingface":
        deploy_model_to_hf_hub()
    elif target == "vllm":
        vllm_model_deployer_step()
    else:
        raise ValueError(f"Invalid target: {target}")
