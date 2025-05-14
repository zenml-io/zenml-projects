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

from src.constants import (
    MAX_MODEL_VERSIONS,
    MODAL_APPROVALS_DIR,
    MODAL_COMPLIANCE_DIR,
    MODAL_DEPLOYMENTS_DIR,
    MODAL_ENVIRONMENT,
    MODAL_EVAL_RESULTS_DIR,
    MODAL_FAIRNESS_DIR,
    MODAL_MODELS_DIR,
    MODAL_MONITORING_DIR,
    MODAL_PREPROCESS_PIPELINE_PATH,
    MODAL_REPORTS_DIR,
    MODAL_RISK_REGISTER_PATH,
    MODAL_VOLUME_NAME,
)


def save_model_with_retention(model: Any) -> Dict[str, Any]:
    """Save model with version control for compliance.

    Args:
        model: The model to save.
        base_dir: The base directory to save the model.
        max_versions: The maximum number of model versions to keep.
        overwrite: Whether to overwrite the model if it already exists.
    """
    # Write new model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"model_{ts}.pkl"
    remote_path = f"{MODAL_MODELS_DIR}/{fname}"

    # Dump locally
    tmp = Path("/tmp") / fname
    joblib.dump(model, tmp)

    save_artifact_to_modal(
        artifact=model,
        artifact_path=remote_path,
        overwrite=False,
    )

    checksum = hashlib.sha256(tmp.read_bytes()).hexdigest()

    vol = Volume.from_name(
        MODAL_VOLUME_NAME,
        create_if_missing=True,
        environment_name=MODAL_ENVIRONMENT,
    )

    # Enforce retention
    try:
        entries = vol.listdir(MODAL_MODELS_DIR)
    except AttributeError:
        entries = [p.name for p in vol.walk(MODAL_MODELS_DIR)]

    # Filter and sort
    mods = sorted(f for f in entries if f.startswith("model_") and f.endswith(".pkl"))
    pruned = []
    if len(mods) > MAX_MODEL_VERSIONS:
        for old in mods[:-MAX_MODEL_VERSIONS]:
            old_path = f"{MODAL_MODELS_DIR}/{old}"
            try:
                # Download old model
                temp_file = Path(tempfile.mkdtemp()) / old
                vol.download(old_path, str(temp_file))

                # Extract key metadata
                old_model = joblib.load(temp_file)
                metadata = {
                    "pruned_at": datetime.now().isoformat(),
                    "model_file": old,
                    "hyperparameters": getattr(old_model, "get_params", lambda: {})(),
                    "feature_importance": getattr(old_model, "feature_importances_", None),
                }

                # Save metadata to archive
                archive_path = f"{MODAL_MODELS_DIR}/archive/{old.replace('.pkl', '_metadata.json')}"
                save_artifact_to_modal(
                    artifact=metadata,
                    artifact_path=archive_path,
                )

                # Now remove the original model
                vol.remove_file(old_path)
                pruned.append({"path": old_path, "archived_metadata": archive_path})
            except Exception as e:
                print(f"Error pruning model {old}: {e}")

    return {"path": remote_path, "checksum": checksum, "pruned": pruned}


def save_artifact_to_modal(
    artifact: Any,
    artifact_path: str,
    overwrite: bool = True,
) -> Optional[str]:
    """Save any artifact to a Modal Volume.

    Args:
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
        environment_name=MODAL_ENVIRONMENT,
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


def save_compliance_artifacts_to_modal(artifacts: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Save compliance-related artifacts to the Modal Volume.

    Args:
        artifacts: Dictionary of artifacts to save with their corresponding paths.

    Returns:
        Dictionary of artifact paths and their checksums.
    """
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for artifact_name, artifact_data in artifacts.items():
        if artifact_name == "preprocess_pipeline":
            path = MODAL_PREPROCESS_PIPELINE_PATH
        elif artifact_name == "evaluation_results":
            path = f"{MODAL_EVAL_RESULTS_DIR}/evaluation_results_{timestamp}.json"
        elif artifact_name == "compliance_report":
            path = f"{MODAL_REPORTS_DIR}/compliance_report_{timestamp}.md"
        elif artifact_name == "model_card":
            path = f"{MODAL_COMPLIANCE_DIR}/model_cards/model_card_{timestamp}.json"
        elif artifact_name == "deployment_record":
            path = f"{MODAL_DEPLOYMENTS_DIR}/deployment_{timestamp}.json"
        elif artifact_name == "approval_record":
            path = f"{MODAL_APPROVALS_DIR}/approval_{timestamp}.json"
        elif artifact_name == "risk_register":
            path = MODAL_RISK_REGISTER_PATH
        elif artifact_name == "fairness_report":
            path = f"{MODAL_FAIRNESS_DIR}/fairness_report_{timestamp}.json"
        elif artifact_name == "monitoring_plan":
            path = f"{MODAL_MONITORING_DIR}/monitoring_plan_{timestamp}.json"
        else:
            # Custom path handling for other artifacts
            path = f"{MODAL_COMPLIANCE_DIR}/artifacts/{artifact_name}_{timestamp}{get_extension_for_artifact(artifact_data)}"

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
