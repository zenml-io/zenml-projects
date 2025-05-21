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


from pathlib import Path
from typing import Annotated, Any, Dict, Optional, Tuple

from zenml import get_step_context, log_metadata, step
from zenml.logger import get_logger

from src.constants import (
    ANNEX_IV_PATH_NAME,
    RELEASES_DIR,
    RUN_RELEASE_DIR,
    SAMPLE_INPUTS_PATH,
    TEMPLATES_DIR,
    VOLUME_METADATA,
)
from src.utils.annex_iv import (
    collect_zenml_metadata,
    generate_model_card,
    generate_readme,
    load_and_process_manual_inputs,
    record_log_locations,
    save_evaluation_artifacts,
    write_git_information,
)
from src.utils.storage import save_compliance_artifacts_to_modal
from src.utils.template import render_annex_iv_template

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_annex_iv_documentation(
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Annotated[str, ANNEX_IV_PATH_NAME],
    Annotated[str, RUN_RELEASE_DIR],
]:
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

    # Step 3: Render the Jinja template with metadata and enriched manual inputs
    content = render_annex_iv_template(
        metadata, manual_inputs, Path(TEMPLATES_DIR)
    )

    # Step 4: Save documentation and artifacts
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

    # Step 7: Record log locations for Article 12 compliance
    log_info = record_log_locations(run_release_dir, pipeline.name, run_id)
    if log_info:
        log_metadata(
            metadata={
                "pipeline_logs_uri": log_info["log_uri"],
                "log_metadata_path": str(
                    run_release_dir / "log_metadata.json"
                ),
            }
        )

    # Step 8: Create and save model card for EU AI Act compliance (Article 13)
    generate_model_card(
        run_release_dir=run_release_dir,
        evaluation_results=evaluation_results,
        deployment_info=deployment_info,
        risk_scores=risk_scores,
    )

    return str(md_path), str(run_release_dir)
