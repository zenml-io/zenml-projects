# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
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

import importlib
from pathlib import Path
from typing import Annotated, Any, Dict

from zenml import log_metadata, step
from zenml.logger import get_logger

from src.constants import (
    APPROVED_NAME,
    DEPLOYMENT_INFO_NAME,
    EVALUATION_RESULTS_NAME,
    MODAL_COMPLIANCE_DIR,
    MODAL_ENVIRONMENT,
    MODAL_VOLUME_NAME,
    MODEL_NAME,
    PREPROCESS_PIPELINE_NAME,
)

logger = get_logger(__name__)

DEPLOYMENT_SCRIPT_PATH = Path(__file__).parent.parent.parent / "app" / "modal_deployment.py"


def load_python_module(file_path: str) -> Any:
    """Dynamically load a Python module from a file path."""
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@step(enable_cache=False)
def modal_deployment(
    approved: Annotated[bool, APPROVED_NAME],
    model: Annotated[Any, MODEL_NAME],
    evaluation_results: Annotated[Dict[str, Any], EVALUATION_RESULTS_NAME],
    preprocess_pipeline: Annotated[Any, PREPROCESS_PIPELINE_NAME],
) -> Annotated[Dict[str, Any], DEPLOYMENT_INFO_NAME]:
    """Deploy model with monitoring and incident reporting (Articles 10, 17, 18).

    This step:
    1. Deploys the model to a Modal endpoint
    2. Sets up data drift monitoring (Article 17)
    3. Configures incident reporting webhook (Article 18)
    4. Logs complete deployment metadata for compliance documentation
    5. Saves all relevant artifacts to Modal volume for compliance

    Args:
        approved: Whether deployment was approved by human oversight
        model: The trained model to deploy
        evaluation_results: Model evaluation metrics and fairness analysis
        preprocess_pipeline: The preprocessing pipeline used in training

    Returns:
        Dictionary with deployment information
    """
    if not approved:
        return {"status": "rejected", "reason": "Not approved by human oversight"}

    # Call Modal deployment script
    module = load_python_module(str(DEPLOYMENT_SCRIPT_PATH))

    deployment_record, model_card = module.run_deployment_entrypoint(
        model=model,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
    )

    # Save deployment record and model card to Modal
    from src.utils.modal_utils import save_compliance_artifacts_to_modal

    artifacts = {
        "deployment_record": deployment_record,
        "model_card": model_card,
    }

    # Save all artifacts to Modal volume
    artifact_paths = save_compliance_artifacts_to_modal(artifacts)

    # Enhanced deployment info with Modal paths
    deployment_info = {
        "deployment_record": deployment_record,
        "artifact_paths": artifact_paths,
        "modal_volume": MODAL_VOLUME_NAME,
        "environment": MODAL_ENVIRONMENT,
    }

    # Log metadata for compliance documentation
    log_metadata(metadata=deployment_info)

    return deployment_info
