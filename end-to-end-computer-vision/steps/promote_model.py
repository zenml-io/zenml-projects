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
from typing import Any, Dict

from zenml import get_step_context, step
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

logger = get_logger(__name__)


@step()
def promote_model(
    metrics: Dict[str, Any],
) -> None:
    """Promotes model to production if better than current production model.

    Args:
        metrics: Metrics to compare against the current production model.
    """
    client = Client()
    # Get the model associated with this pipeline
    cur_model_version = get_step_context().model

    try:
        # Get the current production model
        latest_model_version = client.get_model_version(
            model_name_or_id=cur_model_version.name,
            model_version_name_or_number_or_id=ModelStages.PRODUCTION,
        )
    except KeyError:
        logger.info(
            "No `production` model found to compare to, current "
            "model will be promoted by default."
        )
        cur_model_version.set_stage(ModelStages.PRODUCTION, force=True)

    else:
        prod_metrics = latest_model_version.get_artifact(
            "validation_metrics"
        ).load()
        if metrics["metrics/mAP50(B)"] >= prod_metrics["metrics/mAP50(B)"]:
            logger.info(
                "Model promoted to `production` as it outperformed the "
                "current `production` model on `mAP50`."
            )
            cur_model_version.set_stage(ModelStages.PRODUCTION, force=True)
        else:
            logger.info(
                "Model was less performant than the current `production` "
                "model. Model will **not** be promoted."
            )
