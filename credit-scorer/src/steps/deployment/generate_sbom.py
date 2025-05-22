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
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

import pkg_resources
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot5
from packageurl import PackageURL
from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger

from src.constants import Artifacts as A
from src.constants import Directories

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_sbom(
    deployment_info: Annotated[Optional[Dict[str, Any]], A.DEPLOYMENT_INFO],
) -> Annotated[Dict[str, Any], A.SBOM_ARTIFACT]:
    """Generate SBOM using CycloneDX programmatically."""
    run_id = str(get_step_context().pipeline_run.id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_dir = Path(Directories.RELEASES) / run_id
    os.makedirs(release_dir, exist_ok=True)

    # Create CycloneDX BOM
    bom = Bom()

    # Add only direct dependencies
    direct_deps = get_direct_dependencies()
    for dep in direct_deps:
        component = Component(
            name=dep["name"],
            version=dep["version"],
            type=ComponentType.LIBRARY,
            purl=PackageURL(
                type="pypi", name=dep["name"], version=dep["version"]
            ),
        )
        bom.components.add(component)

    # Convert to JSON
    json_output = JsonV1Dot5(bom)
    sbom_json = json.loads(json_output.output_as_string())

    # Save locally only (not needed by Modal app)
    sbom_filename = "sbom.json"
    local_sbom_path = Path(release_dir) / sbom_filename

    with open(local_sbom_path, "w") as f:
        json.dump(sbom_json, f, indent=2)

    # Generate checksum locally
    import hashlib

    with open(local_sbom_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    sbom_artifact = {
        "sbom_data": sbom_json,
        "sbom_path": str(local_sbom_path),
        "checksum": checksum,
        "generation_time": timestamp,
    }

    log_metadata(metadata={A.SBOM_ARTIFACT: sbom_artifact})
    logger.info(
        f"SBOM generation complete. Saved locally at {local_sbom_path}"
    )

    return sbom_artifact


def get_direct_dependencies():
    """Extract direct dependencies from requirements.txt."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return []

    packages = []
    installed_packages = {pkg.key: pkg for pkg in pkg_resources.working_set}

    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pkg_name = (
                    line.split(">=")[0].split("==")[0].split("<")[0].strip()
                )
                pkg_name = pkg_name.lower().replace("_", "-")

                if pkg_name in installed_packages:
                    pkg = installed_packages[pkg_name]
                    packages.append(
                        {
                            "name": pkg.key,
                            "version": pkg.version,
                        }
                    )

    return packages
