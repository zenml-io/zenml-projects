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

from typing import Dict

from zenml import get_step_context, log_model_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def promote_model(
    metrics: Dict[str, float], target_env: str = "production"
) -> None:
    """Promote model to target environment based on metrics.

    Args:
        metrics: Dictionary of metrics.
        target_env: Target environment to promote the model to.
    """
    log_model_metadata(
        model_name="ecb_interest_rate_model",
        metadata={
            "metrics": metrics,
        },
    )

    if metrics["r2"] > 0.8:
        get_step_context().model.set_stage("production", force=True)
        logger.info(f"Model promoted to {target_env}")
