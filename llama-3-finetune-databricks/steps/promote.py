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

from utils.promote_in_model_registry import promote_in_model_registry
from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.utils.cuda_utils import cleanup_gpu_memory

logger = get_logger(__name__)


@step(enable_cache=False)
def promote(
    metric: str = "rouge1",
    target_stage: str = "staging",
) -> None:
    """Promote the model to the target stage.

    If the model does not outperform the base model, it will be skipped.
    If the model does not outperform the model in the target stage, it will be skipped.

    Args:
        metric: The metric to use for promotion.
        target_stage: The target stage to promote to.
    """
    cleanup_gpu_memory(force=True)
    context_model = get_step_context().model
    base_metrics = context_model.load_artifact("base_model_rouge_metrics")
    ft_metrics = context_model.load_artifact("finetuned_model_rouge_metrics")
    staging_metrics = None
    current_version_model_registry_number = None
    try:
        staging_model = Client().get_model_version(
            context_model.name, target_stage
        )
        staging_metrics = staging_model.get_artifact(
            "finetuned_model_rouge_metrics"
        ).load()
        current_version_model_registry_number =  (
            staging_model.run_metadata["model_registry_version"].value
        )
    except KeyError:
        pass

    msg = (
        f"`{metric}` values to compare:\n"
        f"base={base_metrics[metric]*100:.2f}%\n"
        f"finetuned={ft_metrics[metric]*100:.2f}%"
    )
    if staging_metrics:
        msg += f"\nstaging={staging_metrics[metric]*100:.2f}%"
    logger.info(msg)

    if base_metrics[metric] <= ft_metrics[metric]:
        if staging_metrics is not None and (
            staging_metrics[metric] > ft_metrics[metric]
        ):
            logger.info(
                "Skipping promotion: model does not "
                f"outperform the current model in `{target_stage}`."
            )
        else:
            logger.info(f"Promoting model to `{target_stage}`")
            get_step_context().model.set_stage(target_stage, True)
            
            if Client().active_stack.model_registry:
                # Promote in Model Registry
                latest_version_model_registry_number = context_model.run_metadata[
                    "model_registry_version"
                ].value
                if current_version_model_registry_number is None:
                    current_version_model_registry_number = (
                        latest_version_model_registry_number
                    )
                promote_in_model_registry(
                    latest_version=latest_version_model_registry_number,
                    current_version=current_version_model_registry_number,
                    model_name=context_model.name,
                    target_env=target_stage.capitalize(),
                )
    else:
        logger.info(
            "Skipping promotion: model does not outperform the base model."
        )
