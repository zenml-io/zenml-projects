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

import hashlib
import json
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from modal import Volume


def save_model_to_modal(volume_metadata: Dict, model: Any) -> str:
    """Save a model to a Modal Volume.

    Args:
        volume_metadata: Metadata for the Modal Volume.
        model: The model to save.

    Returns:
        The checksum of the model.
    """
    tmp = Path(tempfile.mkdtemp())
    model_file = tmp / "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    vol = Volume.from_name(
        volume_metadata["volume_name"],
        create_if_missing=True,
        environment_name=volume_metadata["environment_name"],
    )

    # Check if files exist in volume before putting them
    files_in_volume = vol.listdir("/models")
    for file_entry in files_in_volume:
        if file_entry.path == volume_metadata["model_path"]:
            print(f"Deleting existing file: {file_entry.path}")
            vol.remove_file(file_entry.path)

    # Upload the new files
    with vol.batch_upload() as batch:
        batch.put_file(str(model_file), volume_metadata["model_path"])

    # Checksum for integrity
    sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()

    return sha256


def save_artifact_to_modal(
    volume_metadata: Dict,
    artifact: Any,
    artifact_path: str,
    overwrite: bool = True,
) -> Optional[str]:
    """Save any artifact to a Modal Volume.

    Args:
        volume_metadata: Metadata for the Modal Volume.
        artifact: The artifact to save.
        artifact_path: Path within the Modal Volume to save the artifact.
        overwrite: Whether to overwrite the artifact if it already exists.

    Returns:
        The checksum of the artifact if applicable.
    """
    tmp = Path(tempfile.mkdtemp())
    tmp_file = tmp / Path(artifact_path).name

    # Handle different artifact types
    if isinstance(artifact, (dict, list)):
        with open(tmp_file, "w") as f:
            json.dump(artifact, f, indent=2)
    elif hasattr(artifact, "__module__") and "sklearn" in artifact.__module__:
        # Assume scikit-learn object or similar
        joblib.dump(artifact, tmp_file)
    elif isinstance(artifact, str):
        with open(tmp_file, "w") as f:
            f.write(artifact)
    else:
        # Default to pickle for other objects
        with open(tmp_file, "wb") as f:
            pickle.dump(artifact, f)

    vol = Volume.from_name(
        volume_metadata["volume_name"],
        create_if_missing=True,
        environment_name=volume_metadata["environment_name"],
    )

    # Check for existing file
    if overwrite:
        try:
            files_in_volume = vol.listdir("/")
            for file_entry in files_in_volume:
                if file_entry.path == artifact_path:
                    print(f"Deleting existing file: {file_entry.path}")
                    vol.remove_file(file_entry.path)
        except Exception as e:
            print(f"Warning: Could not check for existing files: {e}")

    # Upload the file
    with vol.batch_upload() as batch:
        batch.put_file(str(tmp_file), artifact_path)

    # Generate checksum for binary files
    if tmp_file.exists() and tmp_file.suffix not in [".txt", ".md"]:
        return hashlib.sha256(tmp_file.read_bytes()).hexdigest()
    return None


def save_compliance_artifacts_to_modal(
    volume_metadata: Dict, artifacts: Dict[str, Any]
) -> Dict[str, str]:
    """Save compliance-related artifacts to the Modal Volume.

    Args:
        volume_metadata: Metadata for the Modal Volume.
        artifacts: Dictionary of artifacts to save with their corresponding paths.

    Returns:
        Dictionary of artifact paths and their checksums.
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for artifact_name, artifact_data in artifacts.items():
        if artifact_name == "preprocess_pipeline":
            path = volume_metadata.get(
                "preprocess_pipeline_path", f"pipelines/preprocess_pipeline_{timestamp}.pkl"
            )
        elif artifact_name == "evaluation_results":
            path = volume_metadata.get(
                "evaluation_results_path", f"evaluation/evaluation_results_{timestamp}.json"
            )
        elif artifact_name == "compliance_report":
            path = f"{volume_metadata.get('reports_dir', 'compliance/reports')}/compliance_report_{timestamp}.md"
        elif artifact_name == "model_card":
            path = f"{volume_metadata.get('compliance_dir', 'compliance')}/model_cards/model_card_{timestamp}.json"
        elif artifact_name == "deployment_record":
            path = f"{volume_metadata.get('deployment_records_dir', 'compliance/deployment_records')}/deployment_{timestamp}.json"
        elif artifact_name == "approval_record":
            path = f"{volume_metadata.get('approval_records_dir', 'compliance/approval_records')}/approval_{timestamp}.json"
        else:
            # Custom path handling for other artifacts
            path = (
                f"artifacts/{artifact_name}_{timestamp}{get_extension_for_artifact(artifact_data)}"
            )

        checksum = save_artifact_to_modal(volume_metadata, artifact_data, path)
        results[artifact_name] = {"path": path, "checksum": checksum}

    return results


def get_extension_for_artifact(artifact: Any) -> str:
    """Determine the appropriate file extension based on artifact type."""
    if isinstance(artifact, (dict, list)):
        return ".json"
    elif hasattr(artifact, "__module__") and "sklearn" in artifact.__module__:
        return ".pkl"
    elif isinstance(artifact, str):
        # Check if it looks like markdown
        if artifact.startswith("#") or "```" in artifact:
            return ".md"
        return ".txt"
    else:
        return ".pkl"
