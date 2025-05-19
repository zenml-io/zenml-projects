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
from typing import Any, Dict, List

import yaml
from git import Repo
from zenml.client import Client
from zenml.logger import get_logger
from zenml.models.v2.core.pipeline_run import PipelineRunResponseBody

from src.constants import (
    DEPLOYMENT_PIPELINE_NAME,
    FEATURE_ENGINEERING_PIPELINE_NAME,
    TRAINING_PIPELINE_NAME,
)

logger = get_logger(__name__)


def parse_requirements_txt(requirements_path: str = "requirements.txt") -> Dict[str, str]:
    """Parse requirements.txt and extract package versions.

    Args:
        requirements_path: Path to the requirements.txt file

    Returns:
        Dictionary mapping package names to their versions
    """
    frameworks = {}

    try:
        with open(requirements_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse package==version or package>=version patterns
            if "==" in line:
                package, version = line.split("==", 1)
                frameworks[package.strip()] = version.strip()
            elif ">=" in line:
                package, version = line.split(">=", 1)
                frameworks[package.strip()] = f">={version.strip()}"
            elif "~=" in line:
                package, version = line.split("~=", 1)
                frameworks[package.strip()] = f"~={version.strip()}"
            else:
                # Handle cases like "package" without version
                package = line.split()[0]  # Get first word, ignore any comments
                frameworks[package.strip()] = "latest"

        logger.info(f"Parsed {len(frameworks)} frameworks from {requirements_path}")

    except FileNotFoundError:
        logger.warning(f"requirements.txt not found at {requirements_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing requirements.txt: {e}")
        return {}

    return frameworks


def process_manual_inputs_newlines(manual_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Process manual inputs to handle newlines properly for template rendering.

    Args:
        manual_inputs: Dictionary of manual inputs that may contain \n sequences

    Returns:
        Processed dictionary with proper newlines
    """
    processed = {}

    for key, value in manual_inputs.items():
        if isinstance(value, str):
            # Replace literal \n with actual newlines
            processed[key] = value.replace("\\n", "\n")
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            processed[key] = process_manual_inputs_newlines(value)
        elif isinstance(value, list):
            # Process lists that might contain strings
            processed[key] = [
                item.replace("\\n", "\n") if isinstance(item, str) else item for item in value
            ]
        else:
            # Keep other types as-is
            processed[key] = value

    return processed


def load_and_process_manual_inputs(sample_inputs_path: str) -> Dict[str, Any]:
    """Load and process manual inputs from JSON file, including framework auto-population.

    Args:
        sample_inputs_path: Path to the sample inputs JSON file

    Returns:
        Processed manual inputs dictionary
    """
    # Load sample inputs from fixed path
    try:
        with open(sample_inputs_path, "r") as f:
            manual_inputs = json.load(f)
        logger.info(f"Loaded manual inputs from {sample_inputs_path}")
    except Exception as e:
        logger.error(f"Failed to load sample inputs from {sample_inputs_path}: {e}")
        manual_inputs = {}  # Create empty dict as fallback

    # Process newlines in manual inputs
    manual_inputs = process_manual_inputs_newlines(manual_inputs)

    # Auto-populate frameworks from requirements.txt if not already set
    if "frameworks" not in manual_inputs or not manual_inputs["frameworks"]:
        frameworks_from_requirements = parse_requirements_txt()
        if frameworks_from_requirements:
            manual_inputs["frameworks"] = frameworks_from_requirements
            logger.info(
                f"Auto-populated {len(frameworks_from_requirements)} frameworks from requirements.txt"
            )
        else:
            # Fallback to empty dict if requirements.txt parsing failed
            manual_inputs["frameworks"] = {}
            logger.warning("No frameworks found in requirements.txt, using empty frameworks dict")

    return manual_inputs


def collect_zenml_metadata(context) -> Dict[str, Any]:
    """Collect all relevant metadata from ZenML for Annex IV documentation."""
    # 1. Local git provenance
    repo = Repo(search_parent_directories=True)
    commit = repo.head.commit

    # 2. ZenML client & current run
    client = Client()
    run = context.pipeline_run

    # 3. Get stack information
    stack_data = _collect_stack_information(client)

    # 4. Get previous versions for the deployment pipeline
    previous_versions = _collect_previous_versions(client)

    # 5. Top‑level pipeline & run info
    metadata: Dict[str, Any] = {
        "pipeline": {
            "name": run.pipeline.name,
            "id": str(run.pipeline.id),
            "previous_versions": previous_versions,
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
        "stack": stack_data,
        # environment info
        "environment": {
            "python_version": __import__("sys").version,
            "os": __import__("platform").platform(),
        },
    }

    # 6. Loop over your three pipelines and grab last_run & steps
    pipeline_names = [
        FEATURE_ENGINEERING_PIPELINE_NAME,
        TRAINING_PIPELINE_NAME,
        DEPLOYMENT_PIPELINE_NAME,
    ]

    for pipe_name in pipeline_names:
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
            logger.warning(f"Could not fetch last run for pipeline '{pipe_name}': {e}")

    # 7. Collect artifacts for the current run
    try:
        arts = client.list_artifacts(pipeline_name=run.pipeline.name, pipeline_run_id=run.id)
        for art in arts:
            metadata["run"]["artifacts"][art.name] = art.uri
    except Exception:
        pass

    return metadata


def _process_component(component):
    """Process a component to the desired format."""
    try:
        return {
            "id": str(component.id),
            "name": component.name,
            "type": str(component.type),
            "flavor_name": component.flavor_name,
            "integration": getattr(component, "integration", None),
            "logo_url": getattr(component, "logo_url", None),
        }
    except Exception as e:
        return {
            "id": str(getattr(component, "id", "unknown")),
            "name": getattr(component, "name", "Error processing component"),
            "error": str(e),
        }


def _collect_stack_information(client: Client) -> Dict[str, Any]:
    """Collect stack information from ZenML client."""
    try:
        stack = client.active_stack_model
        stack_data = {
            "id": str(stack.id),
            "name": stack.name,
            "created": str(getattr(stack, "created", None)),
            "updated": str(getattr(stack, "updated", None)),
            "user_id": str(stack.user.id) if hasattr(stack, "user") and stack.user else None,
            "description": getattr(stack, "description", None),
            "components": {},
        }
        for comp_type, components in stack.components.items():
            comp_type_str = str(comp_type)
            stack_data["components"][comp_type_str] = [
                _process_component(component) for component in components
            ]
        return stack_data
    except Exception as e:
        logger.warning(f"Failed to get stack information: {e}")
        return {}


def _collect_previous_versions(client: Client) -> List[Dict[str, Any]]:
    """Collect previous versions of the deployment pipeline."""
    previous_versions = []
    try:
        deployment_pipeline = client.get_pipeline(DEPLOYMENT_PIPELINE_NAME)
        # Get all runs for the deployment pipeline, ordered by creation time (newest first)
        deployment_runs = list(client.list_pipeline_runs(pipeline_id=deployment_pipeline.id))

        # Sort by creation time to get chronological order
        deployment_runs.sort(key=lambda x: x.created, reverse=True)

        # Skip the most recent run (current) and get previous runs
        for pipeline_run in deployment_runs[1:11]:  # Get up to 10 previous versions
            previous_versions.append(
                {
                    "run_id": str(pipeline_run.id),
                    "created": pipeline_run.created.strftime("%Y-%m-%d %H:%M:%S")
                    if pipeline_run.created
                    else "Unknown",
                    "status": str(pipeline_run.status)
                    if hasattr(pipeline_run, "status")
                    else "Unknown",
                    "version": pipeline_run.name or f"Run {str(pipeline_run.id)[:8]}",
                }
            )

        logger.info(
            f"Found {len(previous_versions)} previous versions for {DEPLOYMENT_PIPELINE_NAME}"
        )
    except Exception as e:
        logger.warning(f"Failed to get previous versions for {DEPLOYMENT_PIPELINE_NAME}: {e}")

    return previous_versions


def extract_steps_info(run: PipelineRunResponseBody) -> List[Dict[str, Any]]:
    """Extract step information from a pipeline run.

    Args:
        run: The pipeline run to extract step information from

    Returns:
        List of step information dictionaries
    """
    steps_info = []
    for step_name, step_obj in run.steps.items():
        # Get input information
        inputs_info = {}
        if hasattr(step_obj, "inputs"):
            for input_name, input_value in step_obj.inputs.items():
                if hasattr(input_value, "id"):
                    inputs_info[input_name] = input_value.id
                else:
                    inputs_info[input_name] = str(input_value)

        # Get output information
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


def write_git_information(releases_dir: Path) -> None:
    """Write git information to the releases directory."""
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


def save_evaluation_artifacts(
    releases_dir: Path,
    evaluation_results: Dict[str, Any] = None,
    risk_scores: Dict[str, Any] = None,
) -> None:
    """Save evaluation and risk assessment artifacts as YAML files."""
    if evaluation_results:
        (releases_dir / "evaluation_results.yaml").write_text(yaml.dump(evaluation_results))
    if risk_scores:
        (releases_dir / "risk_scores.yaml").write_text(yaml.dump(risk_scores))


def generate_readme(
    releases_dir: Path,
    pipeline_name: str,
    run_id: str,
    md_name: str,
    has_evaluation_results: bool = False,
    has_risk_scores: bool = False,
) -> None:
    """Generate a comprehensive README file for the documentation bundle."""
    readme = releases_dir / "README.md"
    with open(readme, "w") as f:
        f.write(f"# Documentation for {pipeline_name} (Run {run_id})\n\n")

        # Overview section
        f.write("## Overview\n\n")
        f.write(
            f"This directory contains compliance documentation and artifacts generated for pipeline run `{run_id}`.\n"
        )
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Core documentation
        f.write("### Core Documentation\n\n")
        f.write("| File | Description | EU AI Act Articles |\n")
        f.write("| ---- | ----------- | ------------------ |\n")
        f.write(
            f"| [{md_name}]({md_name}) | Annex IV technical documentation | Art. 11 (Technical Docs) |\n"
        )
        f.write(
            "| [git_info.md](git_info.md) | Git commit and repository information | Art. 12 (Record-keeping) |\n"
        )
        f.write(
            "| [sbom.json](sbom.json) | Software Bill of Materials | Art. 11 (Technical Docs), Art. 12 (Record-keeping) |\n"
        )

        # Evaluation and Risk Assessment
        f.write("\n### Evaluation and Risk Assessment\n\n")
        f.write("| File | Description | EU AI Act Articles |\n")
        f.write("| ---- | ----------- | ------------------ |\n")

        if has_evaluation_results:
            f.write(
                "| [evaluation_results.yaml](evaluation_results.yaml) | Model performance metrics and evaluations | Art. 15 (Accuracy), Art. 13 (Transparency) |\n"
            )
        else:
            f.write(
                "| ~~evaluation_results.yaml~~ | *Not generated for this run* | Art. 15 (Accuracy) |\n"
            )

        if has_risk_scores:
            f.write(
                "| [risk_scores.yaml](risk_scores.yaml) | Risk assessment scores and analysis | Art. 9 (Risk Management) |\n"
            )
        else:
            f.write(
                "| ~~risk_scores.yaml~~ | *Not generated for this run* | Art. 9 (Risk Management) |\n"
            )

        f.write(
            "| [whylogs_profile.html](whylogs_profile.html) | Data profiling report | Art. 10 (Data Governance), Art. 15 (Accuracy) |\n"
        )

        # Monitoring and Post-Deployment
        f.write("\n### Monitoring and Post-Deployment\n\n")
        f.write("| File | Description | EU AI Act Articles |\n")
        f.write("| ---- | ----------- | ------------------ |\n")
        f.write(
            "| [monitoring_plan.json](monitoring_plan.json) | Model monitoring configuration | Art. 15 (Accuracy), Art. 16 (Post-market monitoring) |\n"
        )

        # Additional files (dynamically generate based on what's in the directory)
        excluded_files = {
            "README.md",
            md_name,
            "git_info.md",
            "evaluation_results.yaml",
            "risk_scores.yaml",
            "sbom.json",
            "monitoring_plan.json",
            "whylogs_profile.html",
        }
        other_files = [f for f in releases_dir.glob("*") if f.name not in excluded_files]

        if other_files:
            f.write("\n### Additional Files\n\n")
            f.write("| File | Description |\n")
            f.write("| ---- | ----------- |\n")

            for file_path in other_files:
                file_name = file_path.name
                description = _get_file_description(file_name)
                f.write(f"| [{file_name}]({file_name}) | {description} |\n")

        # EU AI Act compliance section
        f.write("\n## EU AI Act Compliance\n\n")
        f.write("This documentation supports compliance with the EU AI Act, particularly:\n\n")
        f.write("- **Article 11**: Technical Documentation requirements\n")
        f.write("- **Article 9**: Risk Management System\n")
        f.write("- **Article 10**: Data Governance\n")
        f.write("- **Article 13**: Transparency and Information Provision\n")
        f.write("- **Article 15**: Accuracy, Robustness and Cybersecurity\n")
        f.write("- **Article 16**: Post-market monitoring\n\n")

        f.write(
            "For a complete mapping of pipeline steps to EU AI Act articles, see the project's [COMPLIANCE.md](../../../COMPLIANCE.md) file.\n"
        )


def _get_file_description(file_name: str) -> str:
    """Generate a description based on file extension."""
    if file_name.endswith((".yaml", ".yml")):
        return "YAML configuration or data file"
    elif file_name.endswith(".json"):
        return "JSON data file"
    elif file_name.endswith(".md"):
        return "Markdown documentation"
    elif file_name.endswith(".pdf"):
        return "PDF documentation"
    else:
        return "Additional artifact"
