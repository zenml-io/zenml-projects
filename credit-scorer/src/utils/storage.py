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
import yaml
from modal import Volume
from whylogs.viz import NotebookProfileVisualizer
from zenml.client import Client
from zenml.logger import get_logger

from src.constants.annotations import Artifacts as A
from src.constants.annotations import Pipelines
from src.constants.config import ModalConfig

logger = get_logger(__name__)


def save_artifact_to_modal(
    artifact: Any,
    artifact_path: str,
    overwrite: bool = True,
    environment: str = ModalConfig.ENVIRONMENT,
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
        ModalConfig.VOLUME_NAME,
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


def save_whylogs_profile(run_release_dir: Path) -> Path:
    """Save the WhyLogs profile to a file.

    Args:
        run_release_dir: The directory to save the WhyLogs profile to.

    Returns:
        The path to the saved WhyLogs profile.
    """
    client = Client()
    # Get latest data profile
    latest_artifact = client.get_artifact_version(A.WHYLOGS_PROFILE)
    latest_profile = latest_artifact.load()

    pipeline = client.get_pipeline(Pipelines.FEATURE_ENGINEERING)
    runs = pipeline.runs  # sorted newest→oldest

    latest_run, prev_run = runs[0], runs[1] if len(runs) > 1 else runs[0]

    # Default to self vs self
    view_type = "Self-reference"

    # Only load “previous” if it’s actually a different run
    if prev_run.id == latest_run.id:
        prev_profile = latest_profile
    else:
        try:
            prev_prof = prev_run.steps["data_profiler"]
            prev_profile = prev_prof.outputs[A.WHYLOGS_PROFILE][0].load()
            view_type = f"Drift comparison with run <code>{prev_run.id}</code>"
        except Exception:
            prev_profile = latest_profile
            view_type = "Self-reference"

    viz = NotebookProfileVisualizer()
    viz.set_profiles(
        target_profile_view=latest_profile,
        reference_profile_view=prev_profile,
    )
    html = viz.summary_drift_report().data

    # Add a header with info about the comparison type
    header = f"""<div style="background-color: #f8f9fa; padding: 10px; margin-bottom: 20px; border-radius: 5px;">
        <h2>WhyLogs Data Profile Report</h2>
        <p><strong>Report type:</strong> {view_type}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>"""

    # Insert the header after the opening <body> tag
    if "<body>" in html:
        html = html.replace("<body>", f"<body>{header}")
    else:
        html = f"{header}{html}"

    output_path = run_release_dir / "whylogs_profile.html"
    output_path.write_text(html)
    logger.info(f"Saved drift report → {output_path}")

    return output_path


def save_evaluation_visualization(run_release_dir: Path) -> Path:
    """Save evaluation visualization."""
    client = Client()

    eval_html = client.get_artifact_version(
        name_id_or_prefix=A.EVAL_VISUALIZATION
    )

    eval_html_path = run_release_dir / "eval_visualization.html"

    materialized_eval_html = eval_html.load()
    eval_html_path.write_text(materialized_eval_html)

    logger.info(f"Saved evaluation visualization → {eval_html_path}")

    return eval_html_path


def save_evaluation_artifacts(
    run_release_dir: Path,
    evaluation_results: Dict[str, Any] = None,
    risk_scores: Dict[str, Any] = None,
) -> None:
    """Save evaluation and risk assessment artifacts as YAML files."""
    from src.utils.compliance.annex_iv import _summarize_evaluation_results

    if evaluation_results:
        # Create summarized version of evaluation results for local storage
        summarized_results = _summarize_evaluation_results(evaluation_results)
        (run_release_dir / "evaluation_results.yaml").write_text(
            yaml.dump(summarized_results)
        )
    if risk_scores:
        (run_release_dir / "risk_scores.yaml").write_text(
            yaml.dump(risk_scores)
        )

    logger.info(f"Saved evaluation artifacts → {run_release_dir}")


def save_visualizations(run_release_dir: Path) -> None:
    """Save WhyLogs and evaluation visualizations."""
    # save visualizations
    save_evaluation_visualization(run_release_dir)
    save_whylogs_profile(run_release_dir)
