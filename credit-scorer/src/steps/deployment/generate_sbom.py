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
from typing import Annotated, Any, Dict, Optional, Tuple

import pkg_resources
from cyclonedx.model.bom import Bom
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot5
from packageurl import PackageURL
from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger
from zenml.types import HTMLString

from src.constants import Artifacts as A
from src.constants import Directories

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_sbom(
    deployment_info: Annotated[Optional[Dict[str, Any]], A.DEPLOYMENT_INFO],
) -> Tuple[
    Annotated[Dict[str, Any], A.SBOM_ARTIFACT],
    Annotated[HTMLString, A.SBOM_HTML],
]:
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

    # Generate HTML representation of SBOM
    sbom_html = generate_sbom_html(sbom_json, timestamp)

    log_metadata(metadata={A.SBOM_ARTIFACT: sbom_artifact})
    logger.info(
        f"SBOM generation complete. Saved locally at {local_sbom_path}"
    )

    return sbom_artifact, HTMLString(sbom_html)


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


def generate_sbom_html(sbom_data: Dict[str, Any], timestamp: str) -> str:
    """Generate HTML representation of SBOM data."""
    components = sbom_data.get("components", [])
    metadata = sbom_data.get("metadata", {})
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Software Bill of Materials (SBOM)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .metadata {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .checksum {{ font-family: monospace; word-break: break-all; }}
            .purl {{ font-family: monospace; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Software Bill of Materials (SBOM)</h1>
            <p><strong>Format:</strong> {sbom_data.get('bomFormat', 'CycloneDX')}</p>
            <p><strong>Spec Version:</strong> {sbom_data.get('specVersion', 'N/A')}</p>
            <p><strong>Serial Number:</strong> <span class="checksum">{sbom_data.get('serialNumber', 'N/A')}</span></p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>
        
        <div class="metadata">
            <h2>Metadata</h2>
            <p><strong>Timestamp:</strong> {metadata.get('timestamp', 'N/A')}</p>
            <p><strong>Total Components:</strong> {len(components)}</p>
        </div>
        
        <h2>Components ({len(components)} total)</h2>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Version</th>
                    <th>Type</th>
                    <th>Package URL</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for component in sorted(components, key=lambda x: x.get('name', '')):
        name = component.get('name', 'Unknown')
        version = component.get('version', 'Unknown')
        comp_type = component.get('type', 'Unknown')
        purl = component.get('purl', '')
        
        html += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{version}</td>
                    <td>{comp_type}</td>
                    <td class="purl">{purl}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        
        <div style="margin-top: 30px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
            <h3>About this SBOM</h3>
            <p>This Software Bill of Materials (SBOM) was automatically generated as part of EU AI Act compliance requirements (Article 15 - Accuracy & Robustness). It provides a comprehensive inventory of all software components used in the credit scoring model deployment.</p>
        </div>
    </body>
    </html>
    """
    
    return html
