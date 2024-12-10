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
from typing import Optional

import bentoml
from bentoml import bentos
from bentoml._internal.bento import bento
from typing_extensions import Annotated
from zenml import ArtifactConfig, Model, get_step_context, step
from zenml import __version__ as zenml_version
from zenml.client import Client
from zenml.integrations.bentoml.constants import DEFAULT_BENTO_FILENAME
from zenml.integrations.bentoml.steps import bento_builder_step
from zenml.logger import get_logger
from zenml.utils import source_utils

logger = get_logger(__name__)

@step(enable_cache=False)
def bento_dockerizer() -> (
    Annotated[
        str,
        ArtifactConfig(name="bentoml_model_image"),
    ]
):
    """dockerize_bento step.
    
    This step is responsible for dockerizing the BentoML model.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    zenml_client = Client()
    model = get_step_context().model
    version_to_deploy = Model(name=model.name)
    bentoml_deployment = zenml_client.get_artifact_version(name_id_or_prefix="bentoml_rag_deployment")
    bento_tag = f'{bentoml_deployment.run_metadata["bento_tag_name"]}:{bentoml_deployment.run_metadata["bento_info_version"]}'
    container_registry = zenml_client.active_stack.container_registry
    assert container_registry, "Container registry is not configured."
    image_name = f"{container_registry.config.uri}/{bento_tag}"
    image_tag = (image_name,)
    try:
        bentoml.container.build(
            bento_tag=bento_tag,
            backend="docker",  # hardcoding docker since container service only supports docker
            image_tag=image_tag,
        )

    except Exception as e:
        logger.error(f"Error containerizing the bento: {e}")
        raise e
    
    container_registry.push_image(image_name)
    ### YOUR CODE ENDS HERE ###
    return image_name