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
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from modal import Volume

from src.constants import (
    MODAL_ENVIRONMENT,
    MODAL_VOLUME_NAME,
    RELEASES_DIR,
)


class UUIDEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle UUIDs."""

    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)


def save_artifact_to_modal(
    artifact: Any,
    artifact_path: str,
    overwrite: bool = True,
    environment: str = MODAL_ENVIRONMENT,
) -> Optional[str]:
    """Save any artifact to a Modal Volume.

    Args:
        artifact: The artifact to save.
        artifact_path: Path within the Modal Volume to save the artifact.
        overwrite: Whether to overwrite the artifact if it already exists.
        environment: The environment to save the artifact to.

    Returns:
        The checksum of the artifact if applicable.
    """
    tmp = Path(tempfile.mkdtemp())
    tmp_file = tmp / Path(artifact_path).name

    # Handle different artifact types
    if isinstance(artifact, (dict, list)):
        with open(tmp_file, "w") as f:
            json.dump(
                artifact,
                f,
                indent=2,
                default=lambda o: str(o),  # Convert UUIDs to strings
            )
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
        MODAL_VOLUME_NAME,
        create_if_missing=True,
        environment_name=environment,
    )

    # Check for existing file
    if overwrite:
        try:
            vol.remove_file(artifact_path)
        except Exception:
            pass

    # Upload the file
    with vol.batch_upload() as batch:
        batch.put_file(str(tmp_file), artifact_path)

    # Generate checksum for binary files
    if tmp_file.exists() and tmp_file.suffix not in [".txt", ".md"]:
        return hashlib.sha256(tmp_file.read_bytes()).hexdigest()
    return None


def save_compliance_artifacts_to_modal(
    artifacts: Dict[str, Any],
    run_id: str,
) -> Dict[str, Dict[str, str]]:
    """Save compliance-related artifacts to the Modal Volume.

    Args:
        artifacts: Dictionary of artifacts to save with their corresponding paths.
        run_id: Optional pipeline run ID. If not provided, a new UUID will be generated.

    Returns:
        Dictionary of artifact paths and their checksums.
    """
    results = {}

    release_dir = Path(RELEASES_DIR) / run_id
    Path(release_dir).mkdir(parents=True, exist_ok=True)

    for artifact_name, artifact_data in artifacts.items():
        extension = get_extension_for_artifact(artifact_data)
        path = f"{release_dir}/{artifact_name}{extension}"

        checksum = save_artifact_to_modal(
            artifact=artifact_data,
            artifact_path=path,
        )
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
