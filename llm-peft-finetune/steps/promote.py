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

from zenml import get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def promote(metric: str = "rouge1", target_stage: str = "staging") -> None:
    base_metrics = get_step_context().model.load_artifact(
        "base_model_rouge_metrics"
    )
    ft_metrics = get_step_context().model.load_artifact(
        "finetuned_model_rouge_metrics"
    )

    logger.info(
        f"`{metric}` to compare:\nbase={base_metrics[metric]*100:.2f}%\nfinetuned={ft_metrics[metric]*100:.2f}%"
    )
    if base_metrics[metric] <= ft_metrics[metric]:
        logger.info(f"Promoting model to `{target_stage}`")
        get_step_context().model.set_stage(target_stage, True)
    else:
        logger.info("Skipping promotion")
