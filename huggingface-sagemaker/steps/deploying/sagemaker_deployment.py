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
from typing import Optional

from gradio.aws_helper import get_sagemaker_role, get_sagemaker_session
from sagemaker.huggingface import HuggingFaceModel
from typing_extensions import Annotated
from zenml import get_step_context, log_artifact_metadata, step
from zenml.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@step
def deploy_hf_to_sagemaker(
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    transformers_version: str = "4.26.0",
    pytorch_version: str = "1.13.1",
    py_version: str = "py39",
    hf_task: str = "text-classification",
    instance_type: str = "ml.t2.medium",
    container_startup_health_check_timeout: int = 300,
) -> Annotated[str, "sagemaker_endpoint_name"]:
    """
    This step deploy the model to huggingface.

    Args:
        repo_name: The name of the repo to create/use on huggingface.
    """
    # If repo_id and revision are not provided, get them from the model
    #  Otherwise, use the provided values.
    if repo_id is None or revision is None:
        context = get_step_context()
        zenml_model = context.model
        deployment_metadata = zenml_model.get_data_artifact(
            name="huggingface_url"
        ).run_metadata
        repo_id = deployment_metadata["repo_id"]
        revision = deployment_metadata["revision"]

    # Sagemaker
    role = get_sagemaker_role()
    session = get_sagemaker_session()

    hub = {
        "HF_MODEL_ID": repo_id,
        "HF_MODEL_REVISION": revision,
        "HF_TASK": hf_task,
    }

    # Hugging Face Model Class
    huggingface_model = HuggingFaceModel(
        env=hub,
        role=role,  # iam role from AWS
        transformers_version=transformers_version,
        pytorch_version=pytorch_version,
        py_version=py_version,
        sagemaker_session=session,
    )

    # deploy model to SageMaker
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        container_startup_health_check_timeout=container_startup_health_check_timeout,
    )
    endpoint_name = predictor.endpoint_name
    logger.info(f"Model deployed to SageMaker: {endpoint_name}")

    # get region from env variable
    region = os.environ.get("AWS_REGION", "eu-central-1")
    invocation_url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

    log_artifact_metadata(
        artifact_name="sagemaker_endpoint_name",
        metadata={
            "invocation_url": invocation_url,
            "endpoint_name": endpoint_name,
        },
    )


    return endpoint_name
