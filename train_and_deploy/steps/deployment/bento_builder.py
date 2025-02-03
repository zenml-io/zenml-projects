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

from typing_extensions import Annotated

from zenml import ArtifactConfig, get_step_context, step,  __version__ as zenml_version
from zenml.enums import ArtifactType
from zenml.integrations.bentoml.steps import bento_builder_step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils import source_utils
from zenml.integrations.bentoml.constants import DEFAULT_BENTO_FILENAME


import bentoml
from bentoml import bentos
from bentoml._internal.bento import bento

logger = get_logger(__name__)

@step
def bento_builder() -> (
    Annotated[
        Optional[bento.Bento],
        ArtifactConfig(name="mlflow_deployment", artifact_type=ArtifactType.MODEL),
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

        bento_model = bentoml.sklearn.save_model(model.name, model.load_artifact(name="model"))
        # Build the BentoML bundle
        bento = bentos.build(
            service="service.py:svc",
            labels={
                "zenml_version": zenml_version,
                "model_name": model.name,
                "model_version": model.version,
                "model_uri": model.get_artifact(name="model").uri,
                "bento_uri": os.path.join(get_step_context().get_output_artifact_uri(), DEFAULT_BENTO_FILENAME),
            },
            build_ctx=source_utils.get_source_root(),
        )
    else:
        logger.warning("Skipping deployment as the orchestrator is not local.")
        bento = None
    ### YOUR CODE ENDS HERE ###
    return bento

