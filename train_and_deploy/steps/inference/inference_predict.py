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


from typing import Optional, cast
from zenml.client import Client
import pandas as pd
from typing_extensions import Annotated
from zenml import get_step_context, step
from zenml.integrations.bentoml.services.bentoml_local_deployment import (
    BentoMLLocalDeploymentService,
)
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def inference_predict(
    dataset_inf: pd.DataFrame,
    target_env: str,
) -> Annotated[pd.Series, "predictions"]:
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
    model = get_step_context().model

    # get predictor
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        model_name=model.name,
        model_version=target_env,
    )
    predictor_service = cast(BentoMLLocalDeploymentService, existing_services[0])
    if predictor_service is not None:
        # run prediction from service
        predictions = predictor_service.predict(api_endpoint="predict",data=dataset_inf)
    else:
        logger.warning(
            "Predicting from loaded model instead of deployment service "
            "as the orchestrator is not local."
        )
        # run prediction from memory
        predictor = model.load_artifact("model")
        predictions = predictor.predict(dataset_inf)

    predictions = pd.Series(predictions, name="predicted")
    ### YOUR CODE ENDS HERE ###

    return predictions
