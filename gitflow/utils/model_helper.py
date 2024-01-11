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

"""Helper functions used to load models generated and tracked in various stages
of the pipeline."""

from sklearn.base import ClassifierMixin
from typing import Optional, Tuple
from zenml.client import Client
from zenml.enums import ArtifactType
from zenml.services import BaseService
from zenml.post_execution import get_pipeline, StepView


def load_deployed_model(
    model_name: str,
    step_name: str,
) -> Tuple[Optional[BaseService], Optional[ClassifierMixin]]:
    """Load and return the model with the given name being currently served.

    Args:
        model_name: The name of the model to load.
        step_name: The name of the pipeline step that was used to deploy the
            model.
    
    Returns:
        A tuple containing the model deployment service and the loaded model.
        If no model with the given name is currently deployed, the tuple will
        contain None for both values.
    """

    model_deployer = Client().active_stack.model_deployer
    if model_deployer is None:
        raise ValueError("No model deployer found in the active stack.")
    model_servers = model_deployer.find_model_server(model_name=model_name)

    if len(model_servers) == 0:
        print(f"No model with name {model_name} is currently deployed.")
        return None, None

    pipeline_name = model_servers[0].config.pipeline_name
    pipeline_run_id = model_servers[0].config.pipeline_run_id
    # NOTE: this is not accurate as it points to the step function name instead
    # of the pipeline step name. This is a bug in the model deployer.
    # step_name = models[0].config.pipeline_step_name

    pipeline_run = Client().get_pipeline_run(name_id_or_prefix=pipeline_run_id)
    steps_page = Client().list_run_steps(pipeline_run_id=pipeline_run.id)
    step = next((step for step in steps_page.items if step.name == step_name), None)
    if step is None:
        print(
            f"Could not find the pipeline step run with name {step_name} in "
            f"pipeline run {pipeline_run_id} of pipeline {pipeline_name} that "
            f"was used to deploy the model {model_name}."
        )
        return None, None

    step_view = StepView(step)
    step_model_input = step_view.inputs["model"]
    model = step_model_input.read(output_data_type=ClassifierMixin)
    return model_servers[0], model


def load_trained_model(
    pipeline_name: str,
    step_name: str,
    output_name: Optional[str] = None,
) -> Optional[ClassifierMixin]:
    """Load and return the model trained by the last training pipeline run.
    
    Args:
        pipeline_name: The name of the training pipeline.
        step_name: The name of the pipeline step that was used to train the
            model.
        output_name: The name of the output of the pipeline step that was used
            to train the model. If None, the first output of the step will be
            used.
    """

    pipeline = get_pipeline(pipeline_name)
    if pipeline is None:
        raise ValueError(f"No pipeline with name {pipeline_name} found.")
    if len(pipeline.runs) == 0:
        print(f"No pipeline run found for pipeline {pipeline_name}.")
        return None
    pipeline_run = pipeline.runs[0]
    step = pipeline_run.get_step(step_name)
    if step is None:
        print(f"No step with name {step_name} found in pipeline run {pipeline_run.name}.")
        return None
    if not step.is_completed and not step.is_cached:
        print(f"Step {step_name} in pipeline run {pipeline_run.name} is {step.status.value}.")
        return None
    if not step.outputs:
        print(f"Step {step_name} in pipeline run {pipeline_run.name} has no outputs.")
        return None
    if output_name is None:
        output = list(step.outputs.values())[0]
    elif output_name not in step.outputs:
        print(
            f"Step {step_name} in pipeline run {pipeline_run.name} has no output with "
            f"name {output_name}."
        )
        return None
    else:
        output = step.outputs[output_name]
    if output.type != ArtifactType.MODEL.value:
        print(
            f"Output {output.name} of step {step_name} in pipeline run {pipeline_run.name} "
            f"is not a model."
        )
        return None
    model = output.read(output_data_type=ClassifierMixin)
    return model
