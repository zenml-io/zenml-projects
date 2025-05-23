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
import json
from pathlib import Path
from typing import Annotated, Any, Dict

from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger

from src.constants import Artifacts as A
from src.constants import ModalConfig

logger = get_logger(__name__)

DEPLOYMENT_SCRIPT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "modal_app"
    / "modal_deployment.py"
)


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
    approved: Annotated[bool, A.APPROVED],
    approval_record: Annotated[Dict[str, Any], A.APPROVAL_RECORD],
    model: Annotated[Any, A.MODEL],
    evaluation_results: Annotated[Dict[str, Any], A.EVALUATION_RESULTS],
    preprocess_pipeline: Annotated[Any, A.PREPROCESS_PIPELINE],
    environment: str = ModalConfig.ENVIRONMENT,
) -> Annotated[Dict[str, Any], A.DEPLOYMENT_INFO]:
    """Deploy model with monitoring and incident reporting (Articles 10, 17, 18).

    This step:
    1. Deploys the model to a Modal endpoint
    2. Sets up data drift monitoring (Article 17)
    3. Configures incident reporting webhook (Article 18)
    4. Logs complete deployment metadata for compliance documentation
    5. Saves all relevant artifacts to Modal volume for compliance

    Args:
        approved: Whether deployment was approved by human oversight
        approval_record: The approval record for the deployment
        model: The trained model to deploy
        evaluation_results: Model evaluation metrics and fairness analysis
        preprocess_pipeline: The preprocessing pipeline used in training
        environment: The environment to save the artifact to.

    Returns:
        Dictionary with deployment information
    """
    if not approved:
        return {
            "status": "rejected",
            "reason": "Not approved by human oversight",
        }

    # Call Modal deployment script
    module = load_python_module(str(DEPLOYMENT_SCRIPT_PATH))

    deployment_record, model_card = module.run_deployment_entrypoint(
        model=model,
        evaluation_results=evaluation_results,
        preprocess_pipeline=preprocess_pipeline,
    )

    # Add deployment URL to approval record
    deployment_url = deployment_record["endpoints"]["root"]
    approval_record["deployment_url"] = deployment_url

    # Save only essential artifact to Modal
    from src.utils.storage import save_artifact_to_modal

    run_id = str(get_step_context().pipeline_run.id)
    release_dir = Path(f"docs/releases/{run_id}")

    # Save deployment record to Modal as it's needed for API functionality
    deployment_path = f"{release_dir}/deployment_record.json"
    checksum = save_artifact_to_modal(
        artifact=deployment_record,
        artifact_path=deployment_path,
    )

    # Create artifact_paths structure for compatibility
    artifact_paths = {
        "deployment_record": {"path": deployment_path, "checksum": checksum}
    }

    # Make sure release directory exists
    release_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(release_dir / "approval_record.json", "w") as f:
            json.dump(approval_record, f, indent=2, default=str)
        logger.info(
            f"Approval record saved to: {release_dir / 'approval_record.json'}"
        )
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize approval record: {e}")

    # Enhanced deployment info with Modal paths
    deployment_info = {
        "deployment_record": deployment_record,
        "artifact_paths": artifact_paths,
        "modal_volume": ModalConfig.VOLUME_NAME,
        "environment": environment,
    }

    # Log metadata for compliance documentation
    log_metadata(
        metadata={
            "deployment_info": deployment_info,
            "approval_record": approval_record,
        }
    )

    return deployment_info
