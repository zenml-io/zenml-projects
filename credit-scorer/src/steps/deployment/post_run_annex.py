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

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger

from src.constants import (
    ANNEX_IV_PATH_NAME,
    RELEASES_DIR,
    SAMPLE_INPUTS_PATH,
    TEMPLATES_DIR,
    VOLUME_METADATA,
    WHYLOGS_VISUALIZATION_NAME,
)
from src.utils.annex_iv import (
    collect_zenml_metadata,
    generate_readme,
    load_and_process_manual_inputs,
    save_evaluation_artifacts,
    write_git_information,
)
from src.utils.modal_utils import save_compliance_artifacts_to_modal
from src.utils.template import render_annex_iv_template

logger = get_logger(__name__)


def generate_model_card(
    run_release_dir: Path,
    evaluation_results: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a model card for EU AI Act compliance (Article 13).

    Args:
        run_release_dir: Directory path where the model card will be saved
        evaluation_results: Optional evaluation metrics
        deployment_info: Optional deployment information
        risk_scores: Optional risk assessment information

    Returns:
        Path to the generated model card file
    """
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
            "auc": metrics.get("auc", 0.0),
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
        f.write("**Type:** Gradient Boosting Classifier\n")
        f.write("**Framework:** scikit-learn\n\n")

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
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in performance_metrics.items():
                formatted_value = (
                    f"{value:.4f}" if isinstance(value, float) else value
                )
                f.write(f"| {metric} | {formatted_value} |\n")
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


@step(enable_cache=False)
def generate_annex_iv_documentation(
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Annotated[str, ANNEX_IV_PATH_NAME]:
    """Generate Annex IV technical documentation.

    This step implements EU AI Act Annex IV documentation generation
    at the end of a pipeline run.

    Args:
        evaluation_results: Optional evaluation metrics
        risk_scores: Optional risk assessment information
        deployment_info: Optional deployment information from modal_deployment step
        environment: The environment to save the artifact to.

    Returns:
        Path to the generated documentation
    """
    # Get context and setup
    context = get_step_context()
    pipeline_run = context.pipeline_run
    pipeline = context.pipeline
    run_id = str(pipeline_run.id)
    logger.info(f"Generating Annex IV documentation for run: {run_id}")

    # Create immutable releases directory with run_id subdirectory
    run_release_dir = Path(RELEASES_DIR) / run_id
    run_release_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect metadata from context
    metadata = collect_zenml_metadata(context)

    # Add passed artifacts to metadata
    metadata["volume_metadata"] = VOLUME_METADATA
    if evaluation_results:
        metadata["evaluation_results"] = evaluation_results
    if risk_scores:
        metadata["risk_scores"] = risk_scores
    if deployment_info:
        metadata["deployment_info"] = deployment_info

    # Step 2: Load and process manual inputs from sample_inputs.json
    manual_inputs = load_and_process_manual_inputs(SAMPLE_INPUTS_PATH)

    # Step 3: Extract metrics and add deployment info to manual inputs

    # Extract performance metrics from evaluation_results
    if evaluation_results and "metrics" in evaluation_results:
        metrics = evaluation_results["metrics"]
        manual_inputs["performance_metrics"] = {
            "accuracy": metrics.get("accuracy", 0.0),
            "auc": metrics.get("auc", 0.0),
        }
        logger.info(
            f"Extracted performance metrics from evaluation_results: {manual_inputs['performance_metrics']}"
        )

    # Extract fairness metrics from evaluation_results
    if evaluation_results and "fairness" in evaluation_results:
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

        manual_inputs["fairness_assessment"] = fairness_assessment
        logger.info(
            f"Extracted fairness assessment metrics: {len(fairness_assessment)} items"
        )

    # Extract risk information from risk_scores
    if risk_scores:
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

    # Add deployment info to manual inputs if available
    if deployment_info and "deployment_record" in deployment_info:
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
        if "metrics" in record and not manual_inputs.get(
            "performance_metrics"
        ):
            manual_inputs["performance_metrics"] = record["metrics"]
            logger.info(
                f"Using metrics from deployment record: {record['metrics']}"
            )

        logger.info("Added deployment information from deployment_record")

    # Step 4: Render the Jinja template with metadata and enriched manual inputs
    content = render_annex_iv_template(
        metadata, manual_inputs, Path(TEMPLATES_DIR)
    )

    # Step 5: Save documentation and artifacts
    md_name = f"annex_iv_{pipeline.name}.md"
    md_path = run_release_dir / md_name
    md_path.write_text(content)

    # Write additional documentation files
    write_git_information(run_release_dir)
    save_evaluation_artifacts(run_release_dir, evaluation_results, risk_scores)
    generate_readme(
        releases_dir=run_release_dir,
        pipeline_name=pipeline.name,
        run_id=run_id,
        md_name=md_name,
        has_evaluation_results=evaluation_results is not None,
        has_risk_scores=risk_scores is not None,
    )

    # Step 6: Save to Modal volume
    try:
        compliance_artifacts = {
            "compliance_report": content,
            "metadata": metadata,
            "manual_inputs": manual_inputs,
        }

        artifact_paths = save_compliance_artifacts_to_modal(
            compliance_artifacts, run_id
        )
        logger.info(
            f"Compliance artifacts saved to Modal volume: {artifact_paths}"
        )

        # Log the artifact paths to ZenML metadata
        log_metadata(
            metadata={
                "compliance_artifacts": artifact_paths,
                "modal_volume_metadata": VOLUME_METADATA,
                "path": str(md_path),
                "frameworks_count": len(manual_inputs.get("frameworks", {})),
            }
        )

    except Exception as e:
        logger.error(
            f"Failed to save compliance artifacts to Modal volume: {e}"
        )

    # Get whylogs visualization and save to releases directory
    client = Client()
    whylogs_html = client.get_artifact_version(
        name_id_or_prefix=WHYLOGS_VISUALIZATION_NAME
    )

    whylogs_html_path = run_release_dir / "whylogs_profile.html"
    materialized_artifact = whylogs_html.load()

    whylogs_html_path.write_text(materialized_artifact)

    # Record log locations for Article 12 (Record Keeping) compliance
    try:
        # Use the utility function to get log paths without saving them again
        from src.utils.compliance.data_loader import ComplianceDataLoader

        # Get log information for the current pipeline run
        log_info = ComplianceDataLoader.get_pipeline_log_paths(
            pipeline.name, run_id
        )

        if log_info and "log_uri" in log_info:
            # Create pipeline_logs directory for Article 12 compliance requirements
            pipeline_logs_dir = Path(RELEASES_DIR).parent / "pipeline_logs"
            pipeline_logs_dir.mkdir(exist_ok=True, parents=True)

            # Create a JSON file with log metadata
            log_metadata_path = run_release_dir / "log_metadata.json"
            with open(log_metadata_path, "w") as f:
                import json

                json.dump(log_info, f, indent=2)

            # Create a symlink if log file exists and is accessible
            try:
                log_uri = log_info["log_uri"]
                log_file_path = Path(log_uri)
                if log_file_path.exists():
                    # Create a unique name for the symlink
                    symlink_dest = (
                        pipeline_logs_dir / f"{pipeline.name}_{run_id}.log"
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
                        import shutil

                        shutil.copy2(log_file_path, symlink_dest)
                        logger.info(f"Copied log file to {symlink_dest}")

                    # Save the path to the symlink in metadata
                    log_info["pipeline_logs_path"] = str(symlink_dest)
                else:
                    logger.warning(f"Log file not found at {log_uri}")
            except Exception as e:
                logger.warning(f"Failed to create symlink to log file: {e}")

            # Add logs path to ZenML metadata
            log_metadata(
                metadata={
                    "pipeline_logs_uri": log_info["log_uri"],
                    "log_metadata_path": str(log_metadata_path),
                }
            )

            logger.info(
                f"Pipeline logs metadata recorded for Article 12 compliance: {log_info['log_uri']}"
            )
        else:
            logger.warning(f"No log information found for run {run_id}")
    except Exception as e:
        logger.error(f"Failed to record log metadata: {e}")

    # Create and save model card for EU AI Act compliance (Article 13)
    generate_model_card(
        run_release_dir=run_release_dir,
        evaluation_results=evaluation_results,
        deployment_info=deployment_info,
        risk_scores=risk_scores,
    )

    logger.info(f"WhyLogs visualization saved to: {whylogs_html_path}")

    return str(md_path)
