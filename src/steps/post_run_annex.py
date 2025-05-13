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

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import jinja2
import yaml
from git import Repo
from zenml import get_step_context, log_metadata, step
from zenml.client import Client
from zenml.logger import get_logger
from zenml.models.v2.core.pipeline_run import PipelineRunResponseBody

from src.constants import (
    MODAL_VOLUME_NAME,
    VOLUME_METADATA_KEYS,
)

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_annex_iv_documentation(
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
):
    """Generate Annex IV technical documentation.

    This step implements EU AI Act Annex IV documentation generation
    at the end of a pipeline run.

    Args:
        evaluation_results: Optional evaluation metrics
        risk_scores: Optional risk assessment information

    Returns:
        Path to the generated documentation
    """
    # Get client to fetch pipeline run info
    context = get_step_context()
    pipeline_run = context.pipeline_run
    pipeline = context.pipeline

    logger.info(f"Generating Annex IV documentation for run: {pipeline_run.id}")
    logger.info(f"Pipeline name: {pipeline.name}")

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "compliance", "reports")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Collect metadata from context
    metadata = collect_zenml_metadata(context)

    # Add passed artifacts to metadata
    metadata["volume_metadata"] = VOLUME_METADATA_KEYS
    if evaluation_results:
        metadata["evaluation_results"] = evaluation_results
    if risk_scores:
        metadata["risk_scores"] = risk_scores

    # Step 2: Load any manual inputs from YAML files
    manual_inputs = load_manual_inputs(pipeline.name)

    # Step 3: Render the Jinja template
    output_content = render_annex_iv_template(metadata, manual_inputs)

    # Step 4: Save as markdown
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"annex_iv_{pipeline.name}_{timestamp}.md"
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w") as f:
        f.write(output_content)

    logger.info(f"Annex IV documentation saved to: {output_path}")

    # Optional: Convert to PDF
    if os.environ.get("GENERATE_PDF", "0") == "1":
        try:
            from weasyprint import HTML

            pdf_path = output_path.replace(".md", ".pdf")
            HTML(string=output_content).write_pdf(pdf_path)
            logger.info(f"PDF version saved to: {pdf_path}")
        except ImportError:
            logger.warning("WeasyPrint not installed. PDF generation skipped.")

    # Identify missing information
    missing_fields = identify_missing_fields(output_content)
    if missing_fields:
        logger.warning(f"The following fields require manual input: {missing_fields}")

    # Save to Modal volume
    try:
        from src.utils.modal_utils import save_compliance_artifacts_to_modal

        # Save compliance report to Modal volume
        compliance_artifacts = {
            "compliance_report": output_content,
            "metadata": metadata,
            "manual_inputs": manual_inputs,
        }

        # Store artifacts in Modal volume
        artifact_paths = save_compliance_artifacts_to_modal(compliance_artifacts)

        logger.info(f"Compliance artifacts saved to Modal volume: {artifact_paths}")

        # Log the artifact paths to ZenML metadata
        log_metadata(
            metadata={
                "compliance_artifacts": artifact_paths,
                "modal_volume": MODAL_VOLUME_NAME,
            }
        )

    except Exception as e:
        logger.error(f"Failed to save compliance artifacts to Modal volume: {e}")

    return output_path


def collect_zenml_metadata(context) -> Dict[str, Any]:
    """Collect all relevant metadata from ZenML for Annex IV documentation."""
    # Structure the metadata as expected by your template
    repo = Repo(search_parent_directories=True)

    client = Client()
    run = context.pipeline_run
    runs = client.get_pipeline(run.pipeline.name).runs

    metadata = {
        "pipeline": {
            "name": run.pipeline.name,
            "id": run.pipeline.id,
            "description": getattr(run.pipeline, "description", None),
            "previous_versions": [v.id for v in runs] if runs else [],
        },
        "run": {
            "id": run.id,
            "name": run.name,
            # Extract code reference if available
            "code_reference": {
                "commit_sha": repo.head.commit.hexsha,
                "repo_url": next(repo.remotes.origin.urls),
            },
            "metadata": {},  # will be populated below
            "metrics": {},  # will be populated below
            "artifacts": {},  # will be populated below
        },
        "steps": [],
        "environment_variables": os.environ,
        "environment": {"frameworks": {}},
        "stack": None,
    }

    # Get change log
    change_log = repo.git.log("--pretty=format:%h %s", n=10)
    metadata["run"]["metadata"]["change_log"] = change_log.split("\n")

    # Get stack information
    if hasattr(context, "stack") and context.stack:
        metadata["stack"] = {"name": context.stack.name, "components": []}

        if hasattr(context.stack, "components"):
            for component in context.stack.components:
                metadata["stack"]["components"].append(
                    {
                        "name": component.name,
                        "type": component.__class__.__name__,
                        "version": getattr(component, "version", "Unknown"),
                    }
                )

    # Get steps info
    metadata["steps"] = extract_steps_info(run)

    # Get run metadata
    if hasattr(run, "metadata") and run.metadata:
        if isinstance(run.metadata, dict):
            metadata["run"]["metadata"] = run.metadata
        elif hasattr(run.metadata, "__dict__"):
            metadata["run"]["metadata"] = vars(run.metadata)
        else:
            metadata["run"]["metadata"] = {}

    # Get metrics - adjust this based on how metrics are stored in your ZenML instance
    try:
        if hasattr(context, "get_metrics"):
            metadata["run"]["metrics"] = context.get_metrics()
        elif hasattr(run, "metrics"):
            metadata["run"]["metrics"] = run.metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")

    # Get artifacts - adjust this based on how artifacts are stored in your ZenML instance
    try:
        client = context.client if hasattr(context, "client") else None
        if client:
            artifacts = client.list_artifacts(
                pipeline_name=run.pipeline.name, pipeline_run_id=run.id
            )

            artifacts_dict = {}
            artifacts_list = []

            for artifact in artifacts:
                # Add to dictionary format for direct lookup (used in template)
                artifacts_dict[artifact.name] = artifact.uri

                # Add to list format for filtering/iteration in template
                artifacts_list.append(
                    {
                        "name": artifact.name,
                        "type": artifact.type,
                        "uri": artifact.uri,
                        "created_at": artifact.created_at,
                        "version": artifact.version,
                        "metadata": artifact.metadata if hasattr(artifact, "metadata") else {},
                    }
                )

            metadata["run"]["artifacts"] = artifacts_dict
            metadata["artifacts"] = artifacts_list
    except Exception as e:
        logger.error(f"Error fetching artifacts: {e}")

    # Add environment information
    try:
        import platform
        import sys

        metadata["environment"]["python_version"] = sys.version
        metadata["environment"]["os"] = platform.platform()

        try:
            import sklearn

            metadata["environment"]["frameworks"]["scikit-learn"] = sklearn.__version__
        except ImportError:
            pass
    except Exception as e:
        logger.error(f"Error collecting environment info: {e}")

    return metadata


def load_manual_inputs(pipeline_name: str) -> Dict[str, Any]:
    """Load manual inputs from YAML file if it exists."""
    manual_inputs = {}

    manual_inputs_dir = os.path.join(os.getcwd(), "compliance", "manual_fills")
    os.makedirs(manual_inputs_dir, exist_ok=True)

    manual_inputs_path = os.path.join(manual_inputs_dir, f"{pipeline_name}_inputs.yaml")

    if os.path.exists(manual_inputs_path):
        try:
            with open(manual_inputs_path, "r") as f:
                manual_inputs = yaml.safe_load(f)
                logger.info(f"Loaded manual inputs from: {manual_inputs_path}")
        except Exception as e:
            logger.error(f"Error loading manual inputs: {e}")
    else:
        logger.warning(f"No manual inputs file found at: {manual_inputs_path}")
        # Create a template manual inputs file based on what your template expects
        template_inputs = {
            "intended_purpose": "[REQUIRED: Describe the intended purpose of this AI system]",
            "additional_interactions": "[REQUIRED: Describe any interactions with external systems]",
            "design_assumptions": "[REQUIRED: Describe key design choices, including rationale and assumptions]",
            "compliance_tradeoffs": "[REQUIRED: Describe any trade-offs made to comply with Chapter III, Section 2]",
            "computational_resources": "[REQUIRED: Describe computational resources used]",
            "data_methodology": "[REQUIRED: Describe data selection methodology]",
            "oversight_assessment": "[REQUIRED: Provide an assessment of human oversight measures]",
            "continuous_compliance_plan": "[REQUIRED: Describe plans for continuous compliance]",
            "bias_mitigation": "[REQUIRED: Describe bias mitigation measures]",
            "cybersec_measures": "[REQUIRED: Describe cybersecurity measures]",
            "limitations": "[REQUIRED: Describe system limitations]",
            "unintended_outcomes": "[REQUIRED: Describe foreseeable unintended outcomes]",
            "input_specifications": "[REQUIRED: Specify requirements for input data]",
            "metric_appropriateness": "[REQUIRED: Justify chosen performance metrics]",
            "standards_list": "[REQUIRED: List harmonized standards applied]",
            "post_market_plan": "[REQUIRED: Describe post-market monitoring plan]",
        }

        try:
            with open(manual_inputs_path, "w") as f:
                yaml.dump(template_inputs, f, default_flow_style=False)
                logger.info(f"Created template manual inputs file at: {manual_inputs_path}")
        except Exception as e:
            logger.error(f"Error creating template manual inputs file: {e}")

    return manual_inputs


def render_annex_iv_template(metadata: Dict[str, Any], manual_inputs: Dict[str, Any]) -> str:
    """Render the Annex IV Jinja template with collected metadata."""
    # Load template
    template_dir = os.path.join(os.getcwd(), "compliance", "templates")

    loader = jinja2.FileSystemLoader(searchpath=template_dir)
    env = jinja2.Environment(loader=loader)

    # Add custom filters if needed
    env.filters["to_yaml"] = lambda obj: yaml.dump(obj, default_flow_style=False)

    # Your template is already saved as annex_iv_template.j2
    template = env.get_template("annex_iv_template.j2")

    # Set up the template variables as expected by your template
    template_data = {
        "pipeline": metadata["pipeline"],
        "run": metadata["run"],
        "steps": metadata["steps"],
        "artifacts": metadata.get("artifacts", []),
        "environment": metadata["environment"],
        "environment_variables": metadata["environment_variables"],
        "stack": metadata["stack"],
        "manual_inputs": manual_inputs,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Render template
    return template.render(**template_data)


def identify_missing_fields(content: str) -> list:
    """Identify fields that require manual input in the generated document."""
    import re

    missing_fields = []
    # Look for [REQUIRED: ...] and [ORGANIZATION NAME REQUIRED] patterns
    patterns = [r"\[REQUIRED:[^\]]*\]", r"\[ORGANIZATION NAME REQUIRED\]"]

    for pattern in patterns:
        matches = re.findall(pattern, content)
        missing_fields.extend(matches)

    return missing_fields


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
