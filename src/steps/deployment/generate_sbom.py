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

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import pkg_resources
from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger

from src.constants import (
    RELEASES_DIR,
    SBOM_ARTIFACT_NAME,
)
from src.utils.modal_utils import save_artifact_to_modal

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_sbom(
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Annotated[Dict[str, Any], SBOM_ARTIFACT_NAME]:
    """Generate a simplified Software Bill of Materials (SBOM) for Article 15 compliance.

    For this demo, we create a minimal SBOM directly from the Python environment.

    Args:
        deployment_info: Information about the deployed model (optional)

    Returns:
        Dictionary containing SBOM artifact information
    """
    # Get run id as string to avoid JSON serialization issues
    run_id = str(get_step_context().pipeline_run.id)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get release directory for this run
    release_dir = f"{RELEASES_DIR}/{run_id}"

    # Generate SBOM data
    sbom_data = {
        "artifacts": get_packages_info(),
        "source": {
            "type": "python-environment",
            "generated_at": datetime.now().isoformat(),
        },
        "metadata": {
            "description": "Simplified SBOM for AI Act compliance demo",
            "version": "1.0",
            "generation_timestamp": timestamp,
            "pipeline_run_id": run_id,
        },
    }

    # Define file names and paths
    sbom_filename = "sbom.json"
    local_sbom_path = Path(release_dir) / sbom_filename

    with open(local_sbom_path, "w") as f:
        json.dump(sbom_data, f, indent=2)

    # Add deployment info if available
    if deployment_info:
        sbom_data["deployment_info"] = deployment_info

    # Save SBOM directly to Modal volume in the release directory
    modal_sbom_path = f"{release_dir}/{sbom_filename}"
    checksum = save_artifact_to_modal(artifact=sbom_data, artifact_path=modal_sbom_path)

    # Create the artifact object to return
    sbom_artifact = {
        "sbom_data": sbom_data,
        "sbom_path": modal_sbom_path,
        "checksum": checksum,
        "generation_time": timestamp,
    }

    # Log metadata for compliance documentation
    log_metadata(metadata={SBOM_ARTIFACT_NAME: sbom_artifact})

    logger.info(f"SBOM generation complete. Saved to Modal at {modal_sbom_path}")

    return sbom_artifact


def get_packages_info():
    """Get package information from the current Python environment."""
    packages = []
    for pkg in pkg_resources.working_set:
        packages.append(
            {
                "name": pkg.key,
                "version": pkg.version,
                "location": pkg.location,
            }
        )

    return {
        "artifacts": packages,
        "source": {
            "type": "python-environment",
            "generated_at": datetime.now().isoformat(),
        },
        "metadata": {"description": "Simplified SBOM for AI Act compliance demo", "version": "1.0"},
    }
