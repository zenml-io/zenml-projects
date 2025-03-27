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

from typing import Any, Dict, Tuple

from typing_extensions import Annotated
from zenml import get_step_context, step
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def promote_get_metrics() -> (
    Tuple[
        Annotated[Dict[str, Any], "latest_metrics"],
        Annotated[Dict[str, Any], "current_metrics"],
    ]
):
    """Get metrics for comparison for promoting a model.

    This is an example of a metric retrieval step. It is used to retrieve
    a metric from an MLFlow run, that is linked to a model version in the
    model registry. This step is used in the `promote_model` pipeline.

    Args:
        name: Name of the model registered in the model registry.
        metric: Name of the metric to be retrieved.
        version: Version of the model to be retrieved.

    Returns:
        Metric value for a given model.
    """
    ### ADD YOUR OWN CODE HERE - THIS IS JUST AN EXAMPLE ###
    pipeline_extra = get_step_context().pipeline_run.config.extra
    zenml_client = Client()

    # Get current model metric in current run
    current_zenml_model = get_step_context().model
    current_metrics = (
        current_zenml_model.get_model_artifact("model")
        .run_metadata["metrics"]
        .value
    )
    logger.info(f"Current model metrics are {current_metrics}")

    # Get latest saved model version metric in target environment
    try:
        latest_zenml_model = zenml_client.get_model_version(
            model_name_or_id=current_zenml_model.name,
            model_version_name_or_number_or_id=ModelStages(
                pipeline_extra["target_env"]
            ),
        )
    except KeyError:
        latest_zenml_model = None
    if latest_zenml_model:
        latest_metrics = (
            latest_zenml_model.get_model_artifact("model")
            .run_metadata["metrics"]
            .value
        )
        logger.info(f"Current model metrics are {latest_metrics}")
    else:
        logger.info("No currently promoted model found.")
        latest_metrics = current_metrics
    ### YOUR CODE ENDS HERE ###

    return latest_metrics, current_metrics
