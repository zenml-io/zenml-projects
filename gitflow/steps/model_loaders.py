#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
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

"""Model loading steps used to load trained models and deployed models into the
pipeline."""

from typing import List, Optional
from sklearn.base import ClassifierMixin
from zenml.environment import Environment
from zenml.steps import BaseParameters, step
from utils.model_helper import load_deployed_model, load_trained_model
from utils.tracker_helper import get_tracker_name, log_model


class TrainedModelLoaderStepParameters(BaseParameters):
    """Parameters for the trained model loader step.

    Attributes:
        train_pipeline_name: The name of the pipeline that trained the model.
            If not provided, the current pipeline name is used.
        train_pipeline_step_name: The name of the step that trained the model.
        train_pipeline_step_output_name: The name of the output of the step that
            trained the model. If not provided, the first output of the step is
            used.
    """

    train_pipeline_name: Optional[str] = None
    train_pipeline_step_name: str
    train_pipeline_step_output_name: Optional[str] = None


@step(
    enable_cache=False,
    experiment_tracker=get_tracker_name(),
)
def trained_model_loader(
    params: TrainedModelLoaderStepParameters,
) -> ClassifierMixin:
    """Load a trained model from a previous pipeline step.

    Args:
        params: The parameters of the model loader (pipeline name, step name,
            output name).
    
    Returns:
        The loaded model.

    Raises:
        ValueError: If no model is found.
    """
    pipeline_name = (
        params.train_pipeline_name
        or Environment().step_environment.pipeline_name
    )
    model = load_trained_model(
        pipeline_name=pipeline_name,
        step_name=params.train_pipeline_step_name,
        output_name=params.train_pipeline_step_output_name,
    )
    if model:
        # Log the model to the experiment tracker. In the case of the local
        # MLflow tracker, this is needed to serve the model.
        log_model(model, "model")
        return [model]

    raise ValueError("No model found")


class ServedModelLoaderStepParameters(BaseParameters):
    """Parameters for the served model loader step.

    Attributes:
        model_name: The name of the served model to load (i.e. as shown in the
            `zenml model-deployer models list` command).
        step_name: The name of the step that deployed the model in the model
            serving pipeline.
    """

    model_name: str
    step_name: str


@step(enable_cache=False)
def served_model_loader(
    params: ServedModelLoaderStepParameters,
) -> List[ClassifierMixin]:
    """Load a served model.

    This step is used to load a model that is currently served (i.e. listed
    in the `zenml model-deployer models list` command). It uses the information
    provided by the model-deployer to identify the pipeline run that was used to
    deploy the model, then it loads the model from the pipeline run artifacts.

    Args:
        params: The parameters of the model loader (model name, step name).

    Returns:
        The loaded model. If the model is not found, an empty list is returned.
    """
    model_server, model = load_deployed_model(
        model_name=params.model_name,
        step_name=params.step_name,
    )
    if model:
        return [model]

    return []
