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
import importlib
import os
from typing import Optional

import bentoml
from bentoml import bentos
from bentoml._internal.bento import bento
from constants import (
    EMBEDDINGS_MODEL_ID_FINE_TUNED,
)
from typing_extensions import Annotated
from zenml import ArtifactConfig, Model, get_step_context, step
from zenml import __version__ as zenml_version
from zenml.client import Client
from zenml.integrations.bentoml.constants import DEFAULT_BENTO_FILENAME
from zenml.integrations.bentoml.materializers.bentoml_bento_materializer import (
    BentoMaterializer,
)
from zenml.integrations.bentoml.steps import bento_builder_step
from zenml.logger import get_logger
from zenml.orchestrators.utils import get_config_environment_vars
from zenml.utils import source_utils

logger = get_logger(__name__)

@step(output_materializers=BentoMaterializer, enable_cache=False)
def bento_builder() -> (
    Annotated[
        Optional[bento.Bento],
        ArtifactConfig(name="bentoml_rag_deployment", is_model_artifact=True),
    ]
):
    """Predictions step.

    This is an example of a predictions step that takes the data in and returns
    predicted values.

    This step is parameterized, which allows you to configure the step
    independently of the step code, before running it in a pipeline.
    In this example, the step can be configured to use different input data.
    See the documentation for more information:

        https://docs.zenml.io/user-guide/advanced-guide/configure-steps-pipelines

    Args:
        dataset_inf: The inference dataset.

    Returns:
        The predictions as pandas series
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    if Client().active_stack.orchestrator.flavor == "local":
        model = get_step_context().model
        version_to_deploy = Model(name=model.name, version="production")
        logger.info(f"Building BentoML bundle for model: {version_to_deploy.name}")
        # Build the BentoML bundle
        bento = bentos.build(
            service="service.py:RAGService",
            labels={
                "zenml_version": zenml_version,
                "model_name": version_to_deploy.name,
                "model_version": version_to_deploy.version,
                "model_uri": f"zenml/{EMBEDDINGS_MODEL_ID_FINE_TUNED}",
                "bento_uri": os.path.join(get_step_context().get_output_artifact_uri(), DEFAULT_BENTO_FILENAME),
            },
            build_ctx=source_utils.get_source_root(),
            python={
                "requirements_txt":"requirements.txt",
            },
        )
    else:
        logger.warning("Skipping deployment as the orchestrator is not local.")
        bento = None
    ### YOUR CODE ENDS HERE ###
    return bento
