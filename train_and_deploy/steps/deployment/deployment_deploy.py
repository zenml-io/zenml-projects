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


from typing import Optional

from typing_extensions import Annotated

from zenml import ArtifactConfig, get_step_context, step
from zenml.client import Client
from zenml.integrations.bentoml.services.bentoml_deployment import (
    BentoMLDeploymentService,
)
from zenml import Model, log_artifact_metadata
from zenml.integrations.bentoml.steps import bentoml_model_deployer_step
from zenml.logger import get_logger

from bentoml._internal.bento import bento

logger = get_logger(__name__)

@step
def deployment_deploy(
    bento: bento.Bento,
) -> (
    Annotated[
        Optional[BentoMLDeploymentService],
        ArtifactConfig(name="bentoml_deployment", is_deployment_artifact=True),
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

        # deploy predictor service
        bentoml_deployment = bentoml_model_deployer_step.entrypoint(
            model_name=model.name,  # Name of the model
            port=3009,  # Port to be used by the http server
            production=False,  # Deploy the model in production mode
            timeout=1000,
            bento=bento,
        )

        bentoml_service = Client().get_service(name_id_or_prefix=bentoml_deployment.uuid)

        log_artifact_metadata(
            metadata={
                "service_type": "bentoml",
                "status": bentoml_service.state,
                "prediction_url": bentoml_service.prediction_url,
                "health_check_url": bentoml_service.health_check_url,
                "model_uri": model.get_artifact(name="model").uri,
                "bento" : bentoml_service.config.get("bento"),
            }
        )
    else:
        logger.warning("Skipping deployment as the orchestrator is not local.")
        bentoml_deployment = None
    ### YOUR CODE ENDS HERE ###
    return bentoml_deployment