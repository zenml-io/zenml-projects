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
from typing import Annotated, Any, Dict, List, Optional

import jinja2
import yaml
from git import Repo
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.models.v2.core.pipeline_run import PipelineRunResponseBody

from src.constants import (
    ANNEX_IV_PATH_NAME,
    DEPLOYMENT_PIPELINE_NAME,
    FEATURE_ENGINEERING_PIPELINE_NAME,
    MODAL_VOLUME_NAME,
    RELEASES_DIR,
    SAMPLE_INPUTS_PATH,
    TEMPLATES_DIR,
    TRAINING_PIPELINE_NAME,
    VOLUME_METADATA_KEYS,
)

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_annex_iv_documentation(
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
    generate_pdf: Optional[bool] = False,
) -> Annotated[str, ANNEX_IV_PATH_NAME]:
    """Generate Annex IV technical documentation.

    This step implements EU AI Act Annex IV documentation generation
    at the end of a pipeline run.

    Args:
        evaluation_results: Optional evaluation metrics
        risk_scores: Optional risk assessment information
        generate_pdf: Optional flag to generate PDF version

    Returns:
        Path to the generated documentation
    """
    # Get client to fetch pipeline run info
    context = get_step_context()
    pipeline_run = context.pipeline_run
    pipeline = context.pipeline
    run_id = str(pipeline_run.id)
    logger.info(f"Generating Annex IV documentation for run: {run_id}")

    # Create immutable releases directory with run_id subdirectory
    releases_dir = f"{RELEASES_DIR}/{run_id}"
    os.makedirs(releases_dir, exist_ok=True)

    # Step 1: Collect metadata from context
    metadata = collect_zenml_metadata(context)

    # Add passed artifacts to metadata
    metadata["volume_metadata"] = VOLUME_METADATA_KEYS
    if evaluation_results:
        metadata["evaluation_results"] = evaluation_results
    if risk_scores:
        metadata["risk_scores"] = risk_scores

    # Step 2: Load sample inputs from fixed path
    try:
        with open(SAMPLE_INPUTS_PATH, "r") as f:
            manual_inputs = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load sample inputs from {SAMPLE_INPUTS_PATH}: {e}")
        manual_inputs = {}  # Create empty dict as fallback

    # Step 3: Render the Jinja template with the sample inputs
    content = render_annex_iv_template(metadata, manual_inputs)

    # Step 4: Save as markdown to the releases directory
    md_name = f"annex_iv_{pipeline.name}.md"
    md_path = releases_dir / md_name
    md_path.write_text(content)

    # Git provenance
    try:
        repo = Repo(search_parent_directories=True)
        git_md = "# Git Information\n\n"
        git_md += f"**Commit SHA:** {repo.head.commit.hexsha}\n\n"
        git_md += f"**Commit Date:** {datetime.fromtimestamp(repo.head.commit.committed_date).isoformat()}\n\n"
        git_md += (
            f"**Author:** {repo.head.commit.author.name} <{repo.head.commit.author.email}>\n\n"
        )
        git_md += f"**Message:**\n```\n{repo.head.commit.message}\n```\n"
        (releases_dir / "git_info.md").write_text(git_md)
    except Exception:
        logger.warning("Failed to write git info")

    # Optional: Convert to PDF
    if generate_pdf:
        try:
            from weasyprint import HTML

            # Create PDF
            pdf_path = md_path.with_suffix(".pdf")
            HTML(string=content).write_pdf(pdf_path)
            logger.info(f"PDF version saved to: {pdf_path}")
        except ImportError:
            logger.warning("WeasyPrint not installed. PDF generation skipped.")

    # Save evaluation/risk YAML
    if evaluation_results:
        (releases_dir / "evaluation_results.yaml").write_text(yaml.dump(evaluation_results))
    if risk_scores:
        (releases_dir / "risk_scores.yaml").write_text(yaml.dump(risk_scores))

    # README
    readme = releases_dir / "README.md"
    with open(readme, "w") as f:
        f.write(f"# Docs for {pipeline.name} (Run {run_id})\n\n")
        f.write(f"- [Annex IV]({md_name})\n")
        f.write("- [Git info](git_info.md)\n")
        if evaluation_results:
            f.write("- [Evaluation results](evaluation_results.yaml)\n")
        if risk_scores:
            f.write("- [Risk scores](risk_scores.yaml)\n")

    # Modal save
    try:
        from src.utils.modal_utils import save_compliance_artifacts_to_modal

        compliance_artifacts = {
            "compliance_report": content,
            "metadata": metadata,
            "manual_inputs": manual_inputs,
        }

        artifact_paths = save_compliance_artifacts_to_modal(compliance_artifacts, run_id)

        logger.info(f"Compliance artifacts saved to Modal volume: {artifact_paths}")

        # Log the artifact paths to ZenML metadata
        log_metadata(
            metadata={
                "compliance_artifacts": artifact_paths,
                "modal_volume": MODAL_VOLUME_NAME,
                "path": str(md_path),
            }
        )

    except Exception as e:
        logger.error(f"Failed to save compliance artifacts to Modal volume: {e}")

    return str(md_path)


def collect_zenml_metadata(context) -> Dict[str, Any]:
    """Collect all relevant metadata from ZenML for Annex IV documentation."""
    # 1) local git provenance
    repo = Repo(search_parent_directories=True)
    commit = repo.head.commit

    # 2) ZenML client & current run
    client = Client()
    run = context.pipeline_run

    # 3) Top‑level pipeline & run info
    metadata: Dict[str, Any] = {
        "pipeline": {
            "name": run.pipeline.name,
            "id": str(run.pipeline.id),
        },
        "run": {
            "id": str(run.id),
            "name": run.name,
            "code_reference": {
                "commit_sha": commit.hexsha,
                "repo_url": next(repo.remotes.origin.urls, None),
            },
            "metadata": run.metadata if isinstance(run.metadata, dict) else {},
            "metrics": getattr(run, "metrics", {}),
            "artifacts": {},
        },
        "pipelines": [],
        # environment info
        "environment": {
            "python_version": __import__("sys").version,
            "os": __import__("platform").platform(),
        },
    }

    # 4) this pipeline's step info
    metadata["steps"] = extract_steps_info(run)

    # 5) loop over your three pipelines and grab last_run & steps
    for pipe_name in [
        FEATURE_ENGINEERING_PIPELINE_NAME,
        TRAINING_PIPELINE_NAME,
        DEPLOYMENT_PIPELINE_NAME,
    ]:
        try:
            pipe = client.get_pipeline(pipe_name)
            last_run = pipe.last_run
            metadata["pipelines"].append(
                {
                    "name": pipe_name,
                    "run_id": str(last_run.id),
                    "steps": extract_steps_info(last_run),
                }
            )
        except Exception as e:
            # if that pipeline hasn't been run yet, just warn
            import logging

            logging.getLogger(__name__).warning(
                f"Could not fetch last run for pipeline '{pipe_name}': {e}"
            )

    # 6) collect artifacts for the current run (so you can link inputs/outputs by name)
    try:
        arts = client.list_artifacts(pipeline_name=run.pipeline.name, pipeline_run_id=run.id)
        for art in arts:
            metadata["run"]["artifacts"][art.name] = art.uri
    except Exception:
        pass

    return metadata


def render_annex_iv_template(
    metadata: Dict[str, Any],
    manual_inputs: Dict[str, Any],
) -> str:
    """Render the Annex IV Jinja template with collected metadata."""
    # Load template
    template_dir = Path(TEMPLATES_DIR)

    loader = jinja2.FileSystemLoader(searchpath=template_dir)
    env = jinja2.Environment(loader=loader)

    # Add custom filters if needed
    env.filters["to_yaml"] = lambda obj: yaml.dump(obj, default_flow_style=False)

    template = env.get_template("annex_iv_template.j2")

    # Set up the template variables as expected by your template
    template_data = {
        **metadata,
        "manual_inputs": manual_inputs,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Render template
    return template.render(**template_data)


def extract_steps_info(run: PipelineRunResponseBody) -> List[Dict[str, Any]]:
    """Extract step information from a pipeline run.

    Args:
        run: The pipeline run to extract step information from

    Returns:
        List of step information dictionaries
    """
    steps_info = []
    for step_name, step_obj in run.steps.items():
        # Get input and output information
        inputs_info = {}
        outputs_info = {}

        inputs_info = {}
        if hasattr(step_obj, "inputs"):
            for input_name, input_value in step_obj.inputs.items():
                if hasattr(input_value, "id"):
                    inputs_info[input_name] = input_value.id
                else:
                    inputs_info[input_name] = str(input_value)

        # Cleaner version for outputs
        outputs_info = {}
        if hasattr(step_obj, "outputs"):
            for output_name, artifacts in step_obj.outputs.items():
                if artifacts:
                    artifact_ids = [art.id for art in artifacts]
                    outputs_info[output_name] = artifact_ids
                else:
                    outputs_info[output_name] = []

        step_info = {
            "name": step_name,
            "status": step_obj.status if hasattr(step_obj, "status") else None,
            "inputs": inputs_info,
            "outputs": outputs_info,
        }
        steps_info.append(step_info)
    return steps_info
