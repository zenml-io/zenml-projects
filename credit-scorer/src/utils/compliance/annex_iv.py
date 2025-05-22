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
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from git import Repo
from zenml.client import Client
from zenml.logger import get_logger
from zenml.models.v2.core.pipeline_run import PipelineRunResponseBody

from src.constants import Artifacts as A
from src.constants import Directories, Pipelines
from src.utils.compliance.data_loader import ComplianceDataLoader
from src.utils.storage import (
    save_evaluation_visualization,
    save_whylogs_profile,
)

logger = get_logger(__name__)


def parse_requirements_txt(
    requirements_path: str = "requirements.txt",
) -> Dict[str, str]:
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
                package = line.split()[
                    0
                ]  # Get first word, ignore any comments
                frameworks[package.strip()] = "latest"

        logger.info(
            f"Parsed {len(frameworks)} frameworks from {requirements_path}"
        )

    except FileNotFoundError:
        logger.warning(f"requirements.txt not found at {requirements_path}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing requirements.txt: {e}")
        return {}

    return frameworks


def process_manual_inputs_newlines(
    manual_inputs: Dict[str, Any],
) -> Dict[str, Any]:
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
                item.replace("\\n", "\n") if isinstance(item, str) else item
                for item in value
            ]
        else:
            # Keep other types as-is
            processed[key] = value

    return processed


def load_and_process_manual_inputs(
    sample_inputs_path: str,
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and process manual inputs from JSON file, including framework auto-population."""
    # Load sample inputs from fixed path
    try:
        with open(sample_inputs_path, "r") as f:
            manual_inputs = json.load(f)
        logger.info(f"Loaded manual inputs from {sample_inputs_path}")
    except Exception as e:
        logger.error(
            f"Failed to load sample inputs from {sample_inputs_path}: {e}"
        )
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
            logger.warning(
                "No frameworks found in requirements.txt, using empty frameworks dict"
            )

    # Extract performance metrics
    performance_metrics = _extract_performance_metrics(evaluation_results)
    if performance_metrics:
        manual_inputs["performance_metrics"] = performance_metrics

    # Extract fairness assessment
    fairness_assessment = _extract_fairness_assessment(evaluation_results)
    if fairness_assessment:
        manual_inputs["fairness_assessment"] = fairness_assessment

    # Update risk information
    manual_inputs = _update_risk_information(manual_inputs, risk_scores)

    # Extract deployment information
    manual_inputs = _extract_deployment_info(manual_inputs, deployment_info)

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

    metadata["run"]["resource_settings"] = run.config.settings.get(
        "docker", {}
    )

    # 6. Loop over your three pipelines and grab last_run & steps
    pipeline_names = [
        Pipelines.FEATURE_ENGINEERING,
        Pipelines.TRAINING,
        Pipelines.DEPLOYMENT,
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
            logger.warning(
                f"Could not fetch last run for pipeline '{pipe_name}': {e}"
            )

    # 7. Collect artifacts for the current run
    try:
        arts = client.list_artifacts(
            pipeline_name=run.pipeline.name, pipeline_run_id=run.id
        )
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
            "user_id": str(stack.user.id)
            if hasattr(stack, "user") and stack.user
            else None,
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
        deployment_pipeline = client.get_pipeline(Pipelines.DEPLOYMENT)
        # Get all runs for the deployment pipeline, ordered by creation time (newest first)
        deployment_runs = list(
            client.list_pipeline_runs(pipeline_id=deployment_pipeline.id)
        )

        # Sort by creation time to get chronological order
        deployment_runs.sort(key=lambda x: x.created, reverse=True)

        # Skip the most recent run (current) and get previous runs
        for pipeline_run in deployment_runs[
            1:11
        ]:  # Get up to 10 previous versions
            previous_versions.append(
                {
                    "run_id": str(pipeline_run.id),
                    "created": pipeline_run.created.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if pipeline_run.created
                    else "Unknown",
                    "status": str(pipeline_run.status)
                    if hasattr(pipeline_run, "status")
                    else "Unknown",
                    "version": pipeline_run.name
                    or f"Run {str(pipeline_run.id)[:8]}",
                }
            )

        logger.info(
            f"Found {len(previous_versions)} previous versions for {Pipelines.DEPLOYMENT}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to get previous versions for {Pipelines.DEPLOYMENT}: {e}"
        )

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


def write_git_information(run_release_dir: Path) -> None:
    """Write git information to the releases directory."""
    try:
        repo = Repo(search_parent_directories=True)
        git_md = "# Git Information\n\n"
        git_md += f"**Commit SHA:** {repo.head.commit.hexsha}\n\n"
        git_md += f"**Commit Date:** {datetime.fromtimestamp(repo.head.commit.committed_date).isoformat()}\n\n"
        git_md += f"**Author:** {repo.head.commit.author.name} <{repo.head.commit.author.email}>\n\n"
        git_md += f"**Message:**\n```\n{repo.head.commit.message}\n```\n"
        (run_release_dir / "git_info.md").write_text(git_md)
    except Exception:
        logger.warning("Failed to write git info")


def _summarize_evaluation_results(
    evaluation_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize evaluation results to reduce file size while preserving key information."""
    summarized = evaluation_results.copy()

    # Summarize fairness metrics if present
    if (
        "fairness" in summarized
        and "fairness_metrics" in summarized["fairness"]
    ):
        fairness_metrics = summarized["fairness"]["fairness_metrics"]
        summarized_fairness = {}

        for attribute, metrics in fairness_metrics.items():
            if not isinstance(metrics, dict):
                summarized_fairness[attribute] = metrics
                continue

            attr_summary = {}

            # Copy non-group metrics directly
            for key, value in metrics.items():
                if key not in ["accuracy_by_group", "selection_rate_by_group"]:
                    attr_summary[key] = value

            # Helper function to check if groups are numeric
            def is_numeric_groups(groups_dict):
                return all(
                    isinstance(k, (int, float, str))
                    and str(k)
                    .replace(".", "")
                    .replace("-", "")
                    .replace("_", "")
                    .isdigit()
                    for k in groups_dict.keys()
                )

            # Summarize accuracy_by_group if present
            if "accuracy_by_group" in metrics and isinstance(
                metrics["accuracy_by_group"], dict
            ):
                accuracy_groups = metrics["accuracy_by_group"]
                accuracies = list(accuracy_groups.values())

                if (
                    accuracies
                    and is_numeric_groups(accuracy_groups)
                    and len(accuracy_groups) > 10
                ):
                    # For large numeric groups, provide summary statistics
                    attr_summary["accuracy_by_group_summary"] = {
                        "num_groups": len(accuracies),
                        "min_accuracy": round(min(accuracies), 4),
                        "max_accuracy": round(max(accuracies), 4),
                        "mean_accuracy": round(
                            sum(accuracies) / len(accuracies), 4
                        ),
                        "accuracy_range": f"{min(accuracies):.4f} - {max(accuracies):.4f}",
                    }

            summarized_fairness[attribute] = attr_summary

        # Replace the original fairness_metrics with summarized version
        summarized["fairness"]["fairness_metrics"] = summarized_fairness

    return summarized


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
        f.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

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
        other_files = [
            f for f in releases_dir.glob("*") if f.name not in excluded_files
        ]

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
        f.write(
            "This documentation supports compliance with the EU AI Act, particularly:\n\n"
        )
        f.write("- **Article 11**: Technical Documentation requirements\n")
        f.write("- **Article 9**: Risk Management System\n")
        f.write("- **Article 10**: Data Governance\n")
        f.write("- **Article 13**: Transparency and Information Provision\n")
        f.write("- **Article 15**: Accuracy, Robustness and Cybersecurity\n")
        f.write("- **Article 16**: Post-market monitoring\n\n")

        f.write(
            "For a complete mapping of pipeline steps to EU AI Act articles, see the project's [compliance_matrix.md](../../compliance_matrix.md) file.\n"
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


def generate_model_card(
    run_release_dir: Path,
    evaluation_results: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a model card for EU AI Act compliance (Article 13)."""
    model_card_path = run_release_dir / "model_card.md"

    # Basic model information
    model_id = "unknown"
    if deployment_info and "deployment_record" in deployment_info:
        record = deployment_info["deployment_record"]
        model_id = (
            record.get("model_checksum", "")[:8]
            if "model_checksum" in record
            else "unknown"
        )

    model_version = datetime.now().strftime("%Y-%m-%d")
    model_name = "Credit Risk Assessment Model"

    # Performance metrics
    performance_metrics = {}
    if evaluation_results and "metrics" in evaluation_results:
        metrics = evaluation_results["metrics"]
        performance_metrics = {
            "accuracy": metrics.get("accuracy", 0.0),
            "auc": metrics.get("auc_roc", metrics.get("auc", 0.0)),
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "optimal_threshold": metrics.get(
                "optimal_threshold",
                metrics.get("optimal_f1_threshold", 0.5),
            ),
        }

    # Process fairness metrics - summarize instead of showing all groups
    fairness_summary = {}
    if evaluation_results and "fairness" in evaluation_results:
        fairness = evaluation_results["fairness"]
        if "fairness_metrics" in fairness:
            for attribute, metrics in fairness["fairness_metrics"].items():
                attr_summary = {}

                # Add selection rate disparity
                if "selection_rate_disparity" in metrics:
                    attr_summary["selection_rate_disparity"] = metrics[
                        "selection_rate_disparity"
                    ]

                # Summarize accuracy by group instead of showing all values
                if "accuracy_by_group" in metrics and isinstance(
                    metrics["accuracy_by_group"], dict
                ):
                    accuracies = list(metrics["accuracy_by_group"].values())
                    if accuracies:
                        # For numeric groups (like age), show summary stats
                        if all(
                            isinstance(k, (int, float, str))
                            and str(k)
                            .replace(".", "")
                            .replace("-", "")
                            .isdigit()
                            for k in metrics["accuracy_by_group"].keys()
                        ):
                            attr_summary["accuracy_range"] = (
                                f"{min(accuracies):.4f} - {max(accuracies):.4f}"
                            )
                            attr_summary["accuracy_mean"] = (
                                f"{sum(accuracies) / len(accuracies):.4f}"
                            )
                            attr_summary["num_groups"] = len(accuracies)
                        else:
                            # For categorical groups, show the actual groups
                            attr_summary["accuracy_by_group"] = {
                                k: f"{v:.4f}"
                                for k, v in metrics[
                                    "accuracy_by_group"
                                ].items()
                            }

                fairness_summary[attribute] = attr_summary

    # Risk assessment
    overall_risk = 0.5
    if risk_scores and "overall" in risk_scores:
        overall_risk = risk_scores["overall"]

    # Generate the model card content
    with open(model_card_path, "w") as f:
        f.write(f"# {model_name}\n\n")

        # Model Details
        f.write("## Model Details\n\n")
        f.write(f"**Model ID:** {model_id}\n")
        f.write(f"**Version:** {model_version}\n")
        f.write(
            "**Description:** This model assesses credit risk for loan applications\n"
        )
        f.write("**Type:** LGBMClassifier\n")
        f.write("**Framework:** LightGBM\n\n")

        # Intended Use
        f.write("## Intended Use\n\n")
        f.write(
            "This model is designed to assist financial institutions in assessing credit risk for loan applicants. "
        )
        f.write(
            "It predicts the probability of loan default based on applicant financial and demographic data. "
        )
        f.write(
            "The primary use case is to support human decision-makers in loan approval processes.\n\n"
        )

        # Performance Metrics
        f.write("## Performance Metrics\n\n")
        if performance_metrics:
            # Model Quality Metrics (threshold-independent)
            f.write("### Model Quality (Threshold-Independent)\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("|--------|-------|-------------|\n")

            quality_metrics = [
                ("auc_roc", "Area Under ROC Curve"),
                ("average_precision", "Average Precision (PR-AUC)"),
                ("balanced_accuracy", "Balanced Accuracy"),
            ]

            for metric_key, description in quality_metrics:
                if metric_key in performance_metrics:
                    value = performance_metrics[metric_key]
                    formatted_value = (
                        f"{value:.4f}" if isinstance(value, float) else value
                    )
                    f.write(
                        f"| {metric_key} | {formatted_value} | {description} |\n"
                    )

            # Standard Threshold Metrics (0.5)
            f.write("\n### Performance at Standard Threshold (0.5)\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|---------|\n")

            standard_metrics = ["accuracy", "precision", "recall", "f1_score"]

            for metric in standard_metrics:
                if metric in performance_metrics:
                    value = performance_metrics[metric]
                    formatted_value = (
                        f"{value:.4f}" if isinstance(value, float) else value
                    )
                    f.write(f"| {metric} | {formatted_value} |\n")

            # Optimal Threshold Metrics
            f.write(
                f"\n### Performance at Optimal Threshold ({performance_metrics.get('optimal_threshold', 'N/A')})\n\n"
            )
            f.write("| Metric | Value |\n")
            f.write("|--------|---------|\n")

            optimal_metrics = [
                ("optimal_precision", "Precision"),
                ("optimal_recall", "Recall"),
                ("optimal_f1", "F1 Score"),
                ("normalized_cost", "Normalized Cost"),
            ]

            for metric_key, display_name in optimal_metrics:
                if metric_key in performance_metrics:
                    value = performance_metrics[metric_key]
                    formatted_value = (
                        f"{value:.4f}" if isinstance(value, float) else value
                    )
                    f.write(f"| {display_name} | {formatted_value} |\n")

            # Confusion Matrix
            if all(
                k in performance_metrics
                for k in [
                    "true_positives",
                    "false_positives",
                    "true_negatives",
                    "false_negatives",
                ]
            ):
                f.write("\n### Confusion Matrix (at Optimal Threshold)\n\n")
                f.write("| | Predicted: No Default | Predicted: Default |\n")
                f.write("|---|---|---|\n")
                f.write(
                    f"| **Actual: No Default** | {performance_metrics['true_negatives']} (TN) | {performance_metrics['false_positives']} (FP) |\n"
                )
                f.write(
                    f"| **Actual: Default** | {performance_metrics['false_negatives']} (FN) | {performance_metrics['true_positives']} (TP) |\n"
                )
        else:
            f.write("No performance metrics available.\n")
        f.write("\n")

        # Decision Thresholds
        f.write("### Decision Thresholds\n")
        f.write(
            "Model outputs a probability score (0-1). Recommended thresholds:\n"
        )
        f.write("- Low risk: 0.0-0.3\n")
        f.write("- Medium risk: 0.3-0.6\n")
        f.write("- High risk: 0.6-1.0\n\n")

        # Fairness Considerations
        f.write("## Fairness Considerations\n\n")
        f.write(
            "The model has been evaluated for fairness across different demographic groups. "
        )
        f.write(
            "We implement several measures to mitigate potential bias, including:\n"
        )
        f.write("- Protected attributes are not directly used as features\n")
        f.write(
            "- Fairness metrics are evaluated across different demographic groups\n"
        )
        f.write(
            "- Post-processing techniques applied to reduce disparate impact\n\n"
        )

        if fairness_summary:
            f.write("### Fairness Metrics\n\n")
            for attribute, summary in fairness_summary.items():
                f.write(f"#### {attribute.replace('_', ' ').title()}\n\n")

                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")

                for metric_name, metric_value in summary.items():
                    if metric_name == "accuracy_by_group" and isinstance(
                        metric_value, dict
                    ):
                        # Show categorical group accuracies in a separate table
                        f.write("| accuracy_by_group | *see below* |\n")
                    elif isinstance(metric_value, (int, float)):
                        f.write(f"| {metric_name} | {metric_value:.4f} |\n")
                    else:
                        f.write(f"| {metric_name} | {metric_value} |\n")

                # Show categorical accuracy groups if present
                if "accuracy_by_group" in summary and isinstance(
                    summary["accuracy_by_group"], dict
                ):
                    f.write("\n**Accuracy by Group:**\n\n")
                    f.write("| Group | Value |\n")
                    f.write("|-------|-------|\n")
                    for group, acc in summary["accuracy_by_group"].items():
                        f.write(f"| {group} | {acc} |\n")

                f.write("\n")
        else:
            f.write(
                "Detailed fairness metrics are not available for this model version.\n\n"
            )

        # Limitations
        f.write("## Limitations\n\n")
        f.write(
            "- Model performance may degrade when faced with economic conditions significantly different from the training period\n"
        )
        f.write(
            "- Limited validation on certain demographic groups due to data availability\n"
        )
        f.write(
            "- Does not incorporate alternative credit data (utility payments, rent history)\n"
        )
        f.write(
            "- May not generalize well to loan types or amounts significantly different from training distribution\n\n"
        )

        # Risk Management
        f.write("## Risk Management\n\n")
        f.write(
            f"The overall risk score for this model is {overall_risk:.2f} on a scale of 0-1 (lower is better).\n"
        )
        f.write(
            "This model is subject to continuous monitoring for data drift and performance degradation.\n"
        )
        f.write(
            "Human oversight is required for all decisions made with assistance from this model.\n\n"
        )

        # Contact Information
        f.write("## Contact Information\n\n")
        f.write(
            "For questions or concerns about this model, please contact compliance@example.com\n"
        )

    logger.info(f"Generated model card at: {model_card_path}")
    return model_card_path


def _extract_performance_metrics(
    evaluation_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract performance metrics from evaluation results."""
    if not evaluation_results or "metrics" not in evaluation_results:
        return {}

    metrics = evaluation_results["metrics"]
    performance_metrics = {
        # Standard metrics (at default/0.5 threshold)
        "accuracy": metrics.get("accuracy", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1_score": metrics.get("f1_score", 0.0),
        # Model quality metrics
        "auc_roc": metrics.get("auc_roc", metrics.get("auc", 0.0)),
        "average_precision": metrics.get("average_precision", 0.0),
        "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        # Optimal threshold metrics
        "optimal_threshold": metrics.get(
            "optimal_threshold", metrics.get("optimal_cost_threshold", 0.5)
        ),
        "optimal_precision": metrics.get("optimal_precision", 0.0),
        "optimal_recall": metrics.get("optimal_recall", 0.0),
        "optimal_f1": metrics.get("optimal_f1", 0.0),
        # Business impact
        "normalized_cost": metrics.get("normalized_cost", 0.0),
        # Confusion matrix (for transparency)
        "true_positives": metrics.get("true_positives", 0),
        "false_positives": metrics.get("false_positives", 0),
        "true_negatives": metrics.get("true_negatives", 0),
        "false_negatives": metrics.get("false_negatives", 0),
    }

    logger.info(f"Extracted performance metrics: {performance_metrics}")
    return performance_metrics


def _extract_fairness_assessment(
    evaluation_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract fairness assessment from evaluation results."""
    if not evaluation_results or "fairness" not in evaluation_results:
        return {}

    fairness = evaluation_results["fairness"]
    fairness_assessment = {}

    # Extract the bias flag
    if "bias_flag" in fairness:
        fairness_assessment["bias_detected"] = fairness["bias_flag"]

    # Extract fairness metrics for different protected attributes
    if "fairness_metrics" in fairness:
        for attribute, metrics in fairness["fairness_metrics"].items():
            # Handle accuracy disparities
            if "accuracy_by_group" in metrics:
                acc_values = list(metrics["accuracy_by_group"].values())
                if acc_values:
                    max_acc = max(acc_values)
                    min_acc = min(acc_values)
                    fairness_assessment[
                        f"accuracy_disparity_{attribute.lower()}"
                    ] = round(max_acc - min_acc, 3)

            # Handle selection rate disparities
            if "selection_rate_disparity" in metrics:
                fairness_assessment[
                    f"selection_rate_disparity_{attribute.lower()}"
                ] = round(metrics["selection_rate_disparity"], 3)

    # Calculate an overall fairness score if not already present
    if "overall_fairness_score" not in fairness:
        # Simple calculation based on disparities (lower disparity = higher score)
        disparities = [
            v for k, v in fairness_assessment.items() if "disparity" in k
        ]
        if disparities:
            avg_disparity = sum(disparities) / len(disparities)
            fairness_assessment["overall_fairness_score"] = round(
                1.0 - avg_disparity, 2
            )
    else:
        fairness_assessment["overall_fairness_score"] = fairness[
            "overall_fairness_score"
        ]

    logger.info(
        f"Extracted fairness assessment metrics: {len(fairness_assessment)} items"
    )
    return fairness_assessment


def _update_risk_information(
    manual_inputs: Dict[str, Any], risk_scores: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Update risk information with risk scores."""
    if not risk_scores:
        return manual_inputs

    # Update risk management system with actual risk scores if it exists
    risk_system = manual_inputs.get("risk_management_system", "")

    # If we have risk_scores values, update the risk system description
    if "overall" in risk_scores:
        overall_risk = risk_scores["overall"]
        if "0.525" in risk_system:
            risk_system = risk_system.replace("0.525", str(overall_risk))

    if "risk_auc" in risk_scores:
        risk_auc = risk_scores["risk_auc"]
        if "0.25" in risk_system:
            risk_system = risk_system.replace("0.25", str(risk_auc))

    if "risk_bias" in risk_scores:
        risk_bias = risk_scores["risk_bias"]
        if "0.8" in risk_system:
            risk_system = risk_system.replace("0.8", str(risk_bias))

    # If we have hazards information, incorporate it
    if "hazards" in risk_scores:
        hazard_descriptions = [
            h.get("description", "") for h in risk_scores["hazards"]
        ]
        if hazard_descriptions:
            hazard_text = ", ".join(hazard_descriptions)
            manual_inputs["identified_hazards"] = hazard_text

    manual_inputs["risk_management_system"] = risk_system
    logger.info("Updated risk management system with actual risk scores")

    return manual_inputs


def _extract_deployment_info(
    manual_inputs: Dict[str, Any], deployment_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Extract deployment information."""
    if not deployment_info or "deployment_record" not in deployment_info:
        return manual_inputs

    # Extract relevant deployment information
    record = deployment_info["deployment_record"]

    # Update deployment specific fields
    if "endpoints" in record:
        manual_inputs["deployment_endpoints"] = record["endpoints"]

    # Use actual deployment info for the deployment type if available
    if "app_name" in record:
        app_name = record["app_name"]
        manual_inputs["deployment_type"] = (
            f"Modal + FastAPI (Serverless API deployment) - {app_name}"
        )

    # Add any other relevant deployment info to manual_inputs
    manual_inputs["deployment_timestamp"] = record.get("timestamp", "")
    manual_inputs["deployment_id"] = record.get("deployment_id", "")

    # Add model info if available
    if "model_checksum" in record:
        manual_inputs["model_checksum"] = record["model_checksum"]

    # If metrics are available in the deployment record, use them as fallback
    if "metrics" in record and not manual_inputs.get("performance_metrics"):
        manual_inputs["performance_metrics"] = record["metrics"]
        logger.info(
            f"Using metrics from deployment record: {record['metrics']}"
        )

    logger.info("Added deployment information from deployment_record")
    return manual_inputs


def record_log_locations(
    run_release_dir: Path, pipeline_name: str, run_id: str
) -> Dict[str, Any]:
    """Record log locations for Article 12 compliance."""
    try:
        # Get log information for the current pipeline run
        log_info = ComplianceDataLoader.get_pipeline_log_paths(
            pipeline_name, run_id
        )

        if log_info and "log_uri" in log_info:
            # Create pipeline_logs directory for Article 12 compliance requirements
            pipeline_logs_dir = (
                Path(Directories.RELEASES).parent / "pipeline_logs"
            )
            pipeline_logs_dir.mkdir(exist_ok=True, parents=True)

            # Create a JSON file with log metadata
            log_metadata_path = run_release_dir / "log_metadata.json"
            with open(log_metadata_path, "w") as f:
                json.dump(log_info, f, indent=2)

            # Create a symlink if log file exists and is accessible
            try:
                log_uri = log_info["log_uri"]
                log_file_path = Path(log_uri)
                if log_file_path.exists():
                    # Create a unique name for the symlink
                    symlink_dest = (
                        pipeline_logs_dir / f"{pipeline_name}_{run_id}.log"
                    )

                    # Create symlink (or copy file if symlink fails)
                    try:
                        # First remove symlink if it already exists
                        if symlink_dest.exists():
                            symlink_dest.unlink()

                        # Try to create symlink
                        symlink_dest.symlink_to(log_file_path)
                        logger.info(
                            f"Created symlink to log file at {symlink_dest}"
                        )
                    except Exception:
                        # If symlink fails, copy the file
                        shutil.copy2(log_file_path, symlink_dest)
                        logger.info(f"Copied log file to {symlink_dest}")

                    # Save the path to the symlink in metadata
                    log_info["pipeline_logs_path"] = str(symlink_dest)
                else:
                    logger.warning(f"Log file not found at {log_uri}")
            except Exception as e:
                logger.warning(f"Failed to create symlink to log file: {e}")

            logger.info(
                f"Pipeline logs metadata recorded for Article 12 compliance: {log_info['log_uri']}"
            )
            return log_info
        else:
            logger.warning(f"No log information found for run {run_id}")
            return {}
    except Exception as e:
        logger.error(f"Failed to record log metadata: {e}")
        return {}
