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


from zenml import get_step_context, log_metadata, step


@step(enable_cache=False)
def log_metadata_from_step_artifact(
    step_name: str,
    artifact_name: str,
) -> None:
    """Log metadata to the model from saved artifact.

    Args:
        step_name: The name of the step.
        artifact_name: The name of the artifact.
    """

    context = get_step_context()
    # Access the artifact metadata but don't store the unused variable
    _ = context.pipeline_run.steps[step_name].outputs[artifact_name]

    log_metadata(
        artifact_name=artifact_name,
        metadata={"model_name": "phi3.5_finetune_cpu"},
        infer_model=True,
    )
