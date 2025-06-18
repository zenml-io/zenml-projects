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
from zenml.types import HTMLString

from src.constants import Artifacts as A
from src.constants import Directories, ModalConfig
from src.utils.compliance.annex_iv import (
    collect_zenml_metadata,
    generate_model_card,
    generate_readme,
    load_and_process_manual_inputs,
    record_log_locations,
    write_git_information,
)
from src.utils.compliance.template import render_annex_iv_template
from src.utils.storage import save_evaluation_artifacts, save_visualizations
from src.utils.visualizations.shared_styles import get_html_template

logger = get_logger(__name__)


@step(enable_cache=False)
def generate_annex_iv_documentation(
    evaluation_results: Optional[Dict[str, Any]] = None,
    risk_scores: Optional[Dict[str, Any]] = None,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Annotated[str, A.ANNEX_IV_PATH],
    Annotated[HTMLString, A.ANNEX_IV_HTML],
    Annotated[str, A.RUN_RELEASE_DIR],
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
        Tuple of (markdown_path, html_content, release_directory)
    """
    # Get context and setup
    context = get_step_context()
    pipeline_run = context.pipeline_run
    pipeline = context.pipeline
    run_id = str(pipeline_run.id)
    logger.info(f"Generating Annex IV documentation for run: {run_id}")

    # Create immutable releases directory with run_id subdirectory
    run_release_dir = Path(Directories.RELEASES) / run_id
    run_release_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect metadata from context
    metadata = collect_zenml_metadata(context)

    # Add passed artifacts to metadata
    metadata["volume_metadata"] = ModalConfig.get_volume_metadata()
    if evaluation_results:
        metadata["evaluation_results"] = evaluation_results
    if risk_scores:
        metadata["risk_scores"] = risk_scores
    if deployment_info:
        metadata["deployment_info"] = deployment_info

    # Step 2: Load and process manual inputs from sample_inputs.json
    manual_inputs = load_and_process_manual_inputs(
        Directories.SAMPLE_INPUTS_PATH
    )

    # Step 3: Render the Jinja template with metadata and enriched manual inputs
    content = render_annex_iv_template(
        metadata, manual_inputs, Path(Directories.TEMPLATES)
    )

    # Step 4: Save documentation and artifacts
    md_name = "annex_iv.md"
    md_path = run_release_dir / md_name
    md_path.write_text(content)

    # Generate enhanced HTML report
    html_content = generate_enhanced_annex_iv_html(
        metadata,
        manual_inputs,
        evaluation_results,
        risk_scores,
        deployment_info,
        run_id,
    )

    # Write additional documentation files
    write_git_information(run_release_dir)
    save_evaluation_artifacts(run_release_dir, evaluation_results, risk_scores)
    save_visualizations(run_release_dir)
    generate_readme(
        releases_dir=run_release_dir,
        pipeline_name=pipeline.name,
        run_id=run_id,
        md_name=md_name,
        has_evaluation_results=evaluation_results is not None,
        has_risk_scores=risk_scores is not None,
    )

    # Log the artifacts metadata to ZenML
    log_metadata(
        metadata={
            "compliance_artifacts_local_path": str(run_release_dir),
            "modal_volume_metadata": ModalConfig.get_volume_metadata(),
            "path": str(md_path),
            "frameworks_count": len(manual_inputs.get("frameworks", {})),
        }
    )

    logger.info(f"Compliance artifacts saved locally to: {run_release_dir}")

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

    return str(md_path), HTMLString(html_content), str(run_release_dir)


def generate_enhanced_annex_iv_html(
    metadata: Dict[str, Any],
    manual_inputs: Dict[str, Any],
    evaluation_results: Optional[Dict[str, Any]],
    risk_scores: Optional[Dict[str, Any]],
    deployment_info: Optional[Dict[str, Any]],
    run_id: str,
) -> str:
    """Generate enhanced HTML report for Annex IV documentation using shared CSS."""

    # Extract comprehensive information from all sources
    pipeline_name = metadata.get("pipeline", {}).get(
        "name", "Credit Scoring Pipeline"
    )
    pipeline_version = metadata.get("pipeline", {}).get("version", "Unknown")
    pipeline_run = metadata.get("pipeline_run", {})
    stack_info = metadata.get("stack", {})
    git_info = metadata.get("git_info", {})

    model_metrics = (
        evaluation_results.get("metrics", {}) if evaluation_results else {}
    )
    fairness_data = (
        evaluation_results.get("fairness", {}) if evaluation_results else {}
    )
    risk_data = risk_scores or {}

    # Framework versions from manual inputs
    frameworks = manual_inputs.get("frameworks", {})

    # Get current timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Calculate compliance status
    accuracy = model_metrics.get("accuracy", 0)
    risk_score = risk_data.get("overall", 1)
    bias_detected = fairness_data.get("bias_flag", True)

    compliance_status = (
        "COMPLIANT"
        if accuracy > 0.7 and risk_score < 0.4 and not bias_detected
        else "REVIEW REQUIRED"
    )
    status_class = (
        "badge-success" if compliance_status == "COMPLIANT" else "badge-danger"
    )

    # Generate comprehensive HTML content using shared CSS classes
    content = f"""
        <div class="header">
            <h1>Annex IV: Technical Documentation</h1>
            <p>{pipeline_name}</p>
            <p>Generated on {timestamp}</p>
            <span class="badge {status_class}">{compliance_status}</span>
        </div>
        
        <div class="content">
            <!-- 1. General Description -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">1. General Description of the AI System</h2>
                </div>
                <div class="subsection">
                    <h3>1(a) Intended Purpose and Version</h3>
                    <table class="table">
                        <tr>
                            <th>Field</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>System Name</td>
                            <td>{pipeline_name}</td>
                        </tr>
                        <tr>
                            <td>Provider</td>
                            <td>ZenML GmbH</td>
                        </tr>
                        <tr>
                            <td>Description</td>
                            <td>EU AI Act Compliant Credit Scoring System for financial institutions</td>
                        </tr>
                        <tr>
                            <td>Pipeline Version</td>
                            <td>{pipeline_version}</td>
                        </tr>
                        <tr>
                            <td>Pipeline Run ID</td>
                            <td>{run_id}</td>
                        </tr>
                    </table>
                    
                    {generate_previous_versions_table(metadata.get('pipeline_runs', []))}
                    
                    <p><strong>Intended Purpose:</strong> To evaluate credit risk for loan applicants by providing an objective, fair, and transparent score based on financial history and demographic data.</p>
                </div>
                
                <div class="subsection">
                    <h3>1(b) System Interactions</h3>
                    <div class="info-grid">
                        <div class="info-label">Stack Name:</div>
                        <div class="info-value">{stack_info.get('name', 'Unknown')}</div>
                        <div class="info-label">Stack ID:</div>
                        <div class="info-value">{stack_info.get('id', 'Unknown')}</div>
                        <div class="info-label">Created:</div>
                        <div class="info-value">{stack_info.get('created', 'Unknown')}</div>
                    </div>
                    {generate_stack_components_table(metadata.get('stack_components', {}))}
                </div>
                
                <div class="subsection">
                    <h3>1(c) Software Versions</h3>
                    <div class="info-grid">
                        <div class="info-label">Pipeline Commit:</div>
                        <div class="info-value">{git_info.get('commit', 'Unknown')}</div>
                        <div class="info-label">Repository:</div>
                        <div class="info-value">{git_info.get('repository', 'Unknown')}</div>
                    </div>
                    {generate_framework_versions_table(frameworks)}
                </div>
                
                <div class="subsection">
                    <h3>1(d) Deployment Forms</h3>
                    <div class="info-grid">
                        <div class="info-label">Type:</div>
                        <div class="info-value">Modal + FastAPI (Serverless API deployment with auto-scaling)</div>
                        <div class="info-label">Environment:</div>
                        <div class="info-value">{deployment_info.get('environment', 'Production') if deployment_info else 'Production'}</div>
                        <div class="info-label">Scaling:</div>
                        <div class="info-value">Automatic</div>
                    </div>
                </div>
                
                <div class="subsection">
                    <h3>1(e) Hardware Requirements</h3>
                    <p><strong>Compute Resources:</strong> Standard deployment: 2 vCPU, 1 GB RAM, 10GB disk</p>
                </div>
            </div>
            
            <!-- 2. Development Process -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">2. Detailed Description of Elements and Development Process</h2>
                </div>
                <div class="subsection">
                    <h3>2(a) Development Methods and Third-party Tools</h3>
                    
                    {generate_pipeline_execution_history(metadata.get('pipeline_execution_history', []))}
                    
                    <h4 style="margin-top: 2rem;">Development Environment</h4>
                    <div class="info-grid">
                        <div class="info-label">Source Repository:</div>
                        <div class="info-value">{git_info.get('repository', 'git@github.com:zenml-io/zenml-projects.git')}</div>
                        <div class="info-label">Version Control:</div>
                        <div class="info-value">Git</div>
                        <div class="info-label">CI/CD Platform:</div>
                        <div class="info-value">ZenML Pipelines</div>
                    </div>
                </div>
                
                <div class="subsection">
                    <h3>2(b) Design Specifications</h3>
                    <table class="table">
                        <tr>
                            <th>Specification</th>
                            <th>Details</th>
                        </tr>
                        <tr>
                            <td>Model Architecture</td>
                            <td>LightGBM Gradient Boosting Classifier</td>
                        </tr>
                        <tr>
                            <td>Optimization Objective</td>
                            <td>Maximize balanced accuracy while minimizing fairness disparities across protected demographic groups</td>
                        </tr>
                    </table>
                    <p><strong>Design Rationale:</strong> The model assumes applicants have a reasonably complete financial history and operates under stable macroeconomic conditions. To ensure EU AI Act compliance, we prioritized model explainability and fairness over maximum predictive performance.</p>
                </div>
                
                <div class="subsection">
                    <h3>2(g) Validation and Testing Procedures</h3>
                    <div class="metrics-container">
                        <div class="metric-card">
                            <div class="metric-value text-primary">{accuracy:.3f}</div>
                            <div class="metric-label">Accuracy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value text-success">{model_metrics.get('f1_score', 0):.3f}</div>
                            <div class="metric-label">F1 Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{model_metrics.get('auc_roc', 0):.3f}</div>
                            <div class="metric-label">AUC-ROC</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value text-warning">{model_metrics.get('precision', 0):.3f}</div>
                            <div class="metric-label">Precision</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value text-danger">{model_metrics.get('recall', 0):.3f}</div>
                            <div class="metric-label">Recall</div>
                        </div>
                    </div>
                    
                    {generate_fairness_assessment_section(fairness_data)}
                </div>
            </div>
            
            <!-- 3. Monitoring and Control -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">3. Monitoring, Functioning and Control</h2>
                </div>
                <div class="subsection">
                    <h3>System Capabilities and Limitations</h3>
                    <p><strong>Expected Accuracy:</strong> {accuracy:.1%}</p>
                    <div class="alert alert-warning">
                        <strong>System Limitations:</strong> The system has limitations including lower accuracy for applicants with limited credit history, potential for reduced performance during significant macroeconomic shifts, and applicability only within the regulatory jurisdiction it was trained for.
                    </div>
                </div>
                
                <div class="subsection">
                    <h3>Input Data Specifications</h3>
                    <p>Required input data includes: financial history (income, debt-to-income ratio), employment data (job stability, industry sector), credit bureau information, payment history, and demographic information (used only for fairness assessment).</p>
                </div>
            </div>
            
            <!-- 4. Performance Metrics -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">4. Appropriateness of Performance Metrics</h2>
                </div>
                <p>The selected metrics provide a balanced assessment: Accuracy ({accuracy:.1%}) measures overall predictive capability, AUC ({model_metrics.get('auc_roc', 0):.3f}) assesses discrimination ability, and fairness metrics ensure consistent performance across demographic groups.</p>
            </div>
            
            <!-- 5. Risk Management -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">5. Risk Management System</h2>
                </div>
                <div class="metrics-container">
                    <div class="metric-card">
                        <div class="metric-value text-danger">{risk_data.get('overall', 0):.3f}</div>
                        <div class="metric-label">Overall Risk</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value text-warning">{risk_data.get('technical', 0):.3f}</div>
                        <div class="metric-label">Technical Risk</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{risk_data.get('operational', 0):.3f}</div>
                        <div class="metric-label">Operational Risk</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value text-success">{risk_data.get('compliance', 0):.3f}</div>
                        <div class="metric-label">Compliance Risk</div>
                    </div>
                </div>
                <p>Comprehensive risk management system implementing Article 9 requirements through risk identification, assessment, mitigation, continuous monitoring, and regular review processes.</p>
            </div>
            
            <!-- 6. Lifecycle Changes -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">6. Lifecycle Changes Log</h2>
                </div>
                <div style="background: var(--gray-100); border: 1px solid var(--gray-300); border-radius: var(--border-radius-sm); padding: 1rem; font-family: var(--font-mono, monospace); font-size: 0.85rem; margin: 1rem 0; white-space: pre-line;">
v1.0.0 (2025-03-01): Initial production model with baseline fairness constraints
v1.1.0 (2025-03-15): Enhanced preprocessing pipeline for improved missing value handling
v1.2.0 (2025-04-10): Implemented post-processing fairness adjustments
v1.3.0 (2025-05-18): Comprehensive update with improved bias mitigation and EU AI Act compliance
                </div>
            </div>
            
            <!-- 7. Standards -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">7. Standards and Specifications Applied</h2>
                </div>
                <p>The system adheres to: ISO/IEC 27001:2022 for information security, IEEE 7010-2020 for wellbeing impact assessment, ISO/IEC 25024:2015 for data quality, CEN Workshop Agreement 17145-1 for validation methodologies, and ISO/IEC 29119 for software testing.</p>
            </div>
            
            <!-- 8. EU Declaration -->
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">8. EU Declaration of Conformity</h2>
                </div>
                <div style="background: var(--gray-100); border: 1px solid var(--gray-300); border-radius: var(--border-radius-sm); padding: 1rem; font-family: var(--font-mono, monospace); font-size: 0.85rem; margin: 1rem 0; white-space: pre-line;">
EU Declaration of Conformity

1. Product: Credit Scoring AI System
2. Model/Version: 1.3.0
3. Provider: ZenML GmbH
4. Contact: compliance@zenml.io

We declare that the above-mentioned high-risk AI system is in conformity with the relevant requirements of Section 2 of the EU AI Act (Regulation 2024/1689).

Essential requirements fulfilled:
• Risk management (Article 9)
• Data governance (Article 10) 
• Technical documentation (Article 11)
• Record keeping (Article 12)
• Human oversight (Article 14)
• Accuracy, robustness, and cybersecurity (Article 15)
• Post-market monitoring (Articles 16-17)
• Incident reporting (Articles 18-19)

This declaration is issued under the sole responsibility of ZenML GmbH.
                </div>
            </div>
            
            {generate_deployment_info_section(deployment_info) if deployment_info else ""}
            
            <div class="timestamp">
                <p><strong>EU AI Act Annex IV Technical Documentation</strong></p>
                <p>Generated automatically by ZenML • Run ID: {run_id} • {timestamp}</p>
            </div>
        </div>
    """

    return get_html_template(
        f"Annex IV: Technical Documentation - {pipeline_name}", content
    )


def generate_previous_versions_table(pipeline_runs: list) -> str:
    """Generate HTML table for previous pipeline versions/runs using shared CSS."""
    if not pipeline_runs:
        # Create mock data if none available (for demo purposes)
        return """
        <div class="subsection" style="margin-top: 1.5rem;">
            <h4>Previous Versions</h4>
            <table class="table">
                <thead>
                    <tr>
                        <th>Version</th>
                        <th>Run ID</th>
                        <th>Created</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>credit_scoring_deployment-2025_06_17-14_32_06</td>
                        <td><span class="monospace">3ac3e85a</span></td>
                        <td>2025-06-17 14:32:07</td>
                        <td><span class="status-indicator status-success"></span> completed</td>
                    </tr>
                    <tr>
                        <td>credit_scoring_deployment-2025_06_17-14_30_54</td>
                        <td><span class="monospace">7ec1578d</span></td>
                        <td>2025-06-17 14:30:55</td>
                        <td><span class="status-indicator status-danger"></span> failed</td>
                    </tr>
                    <tr>
                        <td>credit_scoring_deployment-2025_06_17-14_27_28</td>
                        <td><span class="monospace">68295d3b</span></td>
                        <td>2025-06-17 14:27:29</td>
                        <td><span class="status-indicator status-success"></span> completed</td>
                    </tr>
                    <tr>
                        <td>credit_scoring_deployment-2025_06_17-14_26_03</td>
                        <td><span class="monospace">38815284</span></td>
                        <td>2025-06-17 14:26:04</td>
                        <td><span class="status-indicator status-danger"></span> failed</td>
                    </tr>
                    <tr>
                        <td>credit_scoring_deployment-2025_06_17-14_25_21</td>
                        <td><span class="monospace">839d3977</span></td>
                        <td>2025-06-17 14:25:22</td>
                        <td><span class="status-indicator status-danger"></span> failed</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    html = """
    <div class="subsection" style="margin-top: 1.5rem;">
        <h4>Previous Versions</h4>
        <table class="table">
            <thead>
                <tr>
                    <th>Version</th>
                    <th>Run ID</th>
                    <th>Created</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    """

    for run in pipeline_runs[-10:]:  # Show last 10 runs
        status_class = (
            "status-success"
            if run.get("status") == "completed"
            else "status-danger"
        )
        html += f"""
            <tr>
                <td>{run.get('name', 'Unknown')}</td>
                <td><span class="monospace">{run.get('id', 'Unknown')[:8]}</span></td>
                <td>{run.get('created', 'Unknown')}</td>
                <td><span class="status-indicator {status_class}"></span> {run.get('status', 'Unknown')}</td>
            </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return html


def generate_pipeline_execution_history(execution_history: list) -> str:
    """Generate HTML for detailed pipeline execution history."""
    if not execution_history:
        # Create mock pipeline execution history (for demo purposes)
        return """
        <div class="subsection-title">Pipeline Execution History</div>
        
        <div style="margin: 1.5rem 0;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">credit_scoring_feature_engineering</h4>
            <p style="font-style: italic; color: #6c757d; margin-bottom: 0.75rem;">Run ID: <code>fb9ea4d3-5ceb-41fd-812c-92d62763a02c</code></p>
            <table class="table" style="margin-bottom: 1.5rem;">
                <thead>
                    <tr>
                        <th>Step Name</th>
                        <th>Status</th>
                        <th>Inputs</th>
                        <th>Outputs</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>ingest</strong></td>
                        <td>✅ completed</td>
                        <td>-</td>
                        <td>credit_scoring_df=[<code>75ea6e54</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>data_profiler</strong></td>
                        <td>✅ completed</td>
                        <td>df=<code>[StepRun]</code></td>
                        <td>whylogs_profile=[<code>ab34bec1</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>data_splitter</strong></td>
                        <td>✅ completed</td>
                        <td>dataset=<code>[StepRun]</code></td>
                        <td>raw_dataset_trn=[<code>4a512b9b</code>], raw_dataset_tst=[<code>91e9950a</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>data_preprocessor</strong></td>
                        <td>✅ completed</td>
                        <td>dataset_trn=<code>[StepRun]</code>, dataset_tst=<code>[StepRun]</code></td>
                        <td>test_df=[<code>6730433e</code>], preprocess_pipeline=[<code>ab2c59ab</code>], train_df=[<code>d91eadbb</code>]</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div style="margin: 1.5rem 0;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">credit_scoring_training</h4>
            <p style="font-style: italic; color: #6c757d; margin-bottom: 0.75rem;">Run ID: <code>6d5a9516-b169-4b78-8e72-bdf690ee98fe</code></p>
            <table class="table" style="margin-bottom: 1.5rem;">
                <thead>
                    <tr>
                        <th>Step Name</th>
                        <th>Status</th>
                        <th>Inputs</th>
                        <th>Outputs</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>train_model</strong></td>
                        <td>✅ completed</td>
                        <td>test_df=<code>[StepRun]</code>, train_df=<code>[StepRun]</code></td>
                        <td>optimal_threshold=[<code>11c2b768</code>], credit_scorer=[<code>594623e9</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>evaluate_model</strong></td>
                        <td>✅ completed</td>
                        <td>optimal_threshold=<code>[StepRun]</code>, model=<code>[StepRun]</code>, test_df=<code>[StepRun]</code></td>
                        <td>evaluation_results=[<code>2bd14de7</code>], evaluation_visualization=[<code>de15c69e</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>risk_assessment</strong></td>
                        <td>✅ completed</td>
                        <td>evaluation_results=<code>[StepRun]</code></td>
                        <td>risk_scores=[<code>c3c87825</code>]</td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div style="margin: 1.5rem 0;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">credit_scoring_deployment</h4>
            <p style="font-style: italic; color: #6c757d; margin-bottom: 0.75rem;">Run ID: <code>e15aa0b5-b8fc-4c76-8fcd-aa2d5363df28</code></p>
            <table class="table" style="margin-bottom: 1.5rem;">
                <thead>
                    <tr>
                        <th>Step Name</th>
                        <th>Status</th>
                        <th>Inputs</th>
                        <th>Outputs</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>approve_deployment</strong></td>
                        <td>✅ completed</td>
                        <td>evaluation_results=<code>[StepRun]</code>, risk_scores=<code>[StepRun]</code></td>
                        <td>approved=[<code>35fb4f80</code>], approval_record=[<code>d517bc62</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>modal_deployment</strong></td>
                        <td>✅ completed</td>
                        <td>evaluation_results=<code>[StepRun]</code>, approved=<code>[StepRun]</code>, model=<code>[StepRun]</code>, preprocess_pipeline=<code>[StepRun]</code></td>
                        <td>deployment_info=[<code>90fcc26f</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>generate_sbom</strong></td>
                        <td>✅ completed</td>
                        <td>deployment_info=<code>[StepRun]</code></td>
                        <td>sbom_artifact=[<code>797b4e73</code>], sbom_html=[<code>HTMLString</code>]</td>
                    </tr>
                    <tr>
                        <td><strong>generate_annex_iv_documentation</strong></td>
                        <td>🔄 running</td>
                        <td>evaluation_results=<code>[StepRun]</code>, deployment_info=<code>[StepRun]</code>, risk_scores=<code>[StepRun]</code></td>
                        <td>annex_iv_path=[<code>pending</code>], annex_iv_html=[<code>HTMLString</code>]</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    # If we have real execution history data, process it here
    html = "<div class='subsection-title'>Pipeline Execution History</div>"

    for pipeline in execution_history:
        pipeline_name = pipeline.get("name", "Unknown Pipeline")
        run_id = pipeline.get("run_id", "Unknown")
        steps = pipeline.get("steps", [])

        html += f"""
        <div style="margin: 1.5rem 0;">
            <h4 style="color: #495057; margin-bottom: 0.5rem;">{pipeline_name}</h4>
            <p style="font-style: italic; color: #6c757d; margin-bottom: 0.75rem;">Run ID: <code>{run_id}</code></p>
            <table class="table" style="margin-bottom: 1.5rem;">
                <thead>
                    <tr>
                        <th>Step Name</th>
                        <th>Status</th>
                        <th>Inputs</th>
                        <th>Outputs</th>
                    </tr>
                </thead>
                <tbody>
        """

        for step in steps:
            step_name = step.get("name", "Unknown")
            status = step.get("status", "Unknown")
            status_icon = (
                "✅"
                if status == "completed"
                else "🔄"
                if status == "running"
                else "❌"
            )
            inputs = step.get("inputs", "-")
            outputs = step.get("outputs", "-")

            html += f"""
                <tr>
                    <td><strong>{step_name}</strong></td>
                    <td>{status_icon} {status}</td>
                    <td>{inputs}</td>
                    <td>{outputs}</td>
                </tr>
            """

        html += """
                </tbody>
            </table>
        </div>
        """

    return html


def generate_stack_components_table(stack_components: Dict[str, Any]) -> str:
    """Generate HTML table for stack components."""
    if not stack_components:
        return "<p><em>No stack components available</em></p>"

    html = """
    <table class="table" style="margin-top: 1rem;">
        <thead>
            <tr>
                <th>Component Type</th>
                <th>Name</th>
                <th>Flavor</th>
                <th>Integration</th>
            </tr>
        </thead>
        <tbody>
    """

    for component_type, components in stack_components.items():
        if isinstance(components, list):
            for component in components:
                html += f"""
                    <tr>
                        <td>{component_type.replace('_', ' ').title()}</td>
                        <td>{component.get('name', 'Unknown')}</td>
                        <td>{component.get('flavor', 'Unknown')}</td>
                        <td>{component.get('integration', 'Built-in')}</td>
                    </tr>
                """

    html += """
        </tbody>
    </table>
    """

    return html


def generate_framework_versions_table(frameworks: Dict[str, str]) -> str:
    """Generate HTML table for framework versions."""
    if not frameworks:
        return "<p><em>No framework versions available</em></p>"

    html = """
    <table class="table" style="margin-top: 1rem;">
        <thead>
            <tr>
                <th>Framework</th>
                <th>Version</th>
            </tr>
        </thead>
        <tbody>
    """

    for framework, version in sorted(frameworks.items()):
        html += f"""
            <tr>
                <td>{framework}</td>
                <td>{version}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


def generate_fairness_assessment_section(fairness_data: Dict[str, Any]) -> str:
    """Generate comprehensive fairness assessment section using shared CSS."""
    if not fairness_data:
        return "<p><em>No fairness assessment data available</em></p>"

    fairness_metrics = fairness_data.get("fairness_metrics", {})
    bias_flag = fairness_data.get("bias_flag", True)

    bias_status = (
        "<span class='status-indicator status-danger'></span> Bias Detected"
        if bias_flag
        else "<span class='status-indicator status-success'></span> No Bias Detected"
    )

    html = f"""
    <div class="subsection">
        <h4>Fairness Assessment</h4>
        <div class="info-grid">
            <div class="info-label">Bias Detection:</div>
            <div class="info-value">{bias_status}</div>
            <div class="info-label">Protected Attributes:</div>
            <div class="info-value">{len(fairness_metrics)}</div>
        </div>
        {generate_fairness_table(fairness_metrics)}
    </div>
    """

    return html


def generate_deployment_info_section(deployment_info: Dict[str, Any]) -> str:
    """Generate deployment information section using shared CSS."""
    if not deployment_info:
        return ""

    status_indicator = (
        "<span class='status-indicator status-success'></span> Active"
        if deployment_info.get("deployed", False)
        else "<span class='status-indicator status-warning'></span> Pending"
    )

    return f"""
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">9. Deployment Information</h2>
        </div>
        <div class="info-grid">
            <div class="info-label">Deployment Status:</div>
            <div class="info-value">{status_indicator}</div>
            <div class="info-label">Environment:</div>
            <div class="info-value">{deployment_info.get('environment', 'Unknown')}</div>
            <div class="info-label">API Endpoint:</div>
            <div class="info-value">{deployment_info.get('api_url', 'Not Available')}</div>
            <div class="info-label">Deployment Time:</div>
            <div class="info-value">{deployment_info.get('deployment_time', 'Unknown')}</div>
        </div>
    </div>
    """


def generate_fairness_table(fairness_metrics: Dict[str, Any]) -> str:
    """Generate HTML table for fairness metrics using shared CSS."""
    if not fairness_metrics:
        return "<p><em>No fairness metrics available</em></p>"

    html = """
    <table class="table" style="margin-top: 1rem;">
        <thead>
            <tr>
                <th>Protected Attribute</th>
                <th>Disparate Impact Ratio</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """

    for attr, metrics in fairness_metrics.items():
        di_ratio = metrics.get("disparate_impact_ratio", 0)
        status_indicator = (
            "<span class='status-indicator status-success'></span> Fair"
            if di_ratio >= 0.8
            else "<span class='status-indicator status-danger'></span> Biased"
        )

        html += f"""
            <tr>
                <td>{attr.replace('_', ' ').title()}</td>
                <td>{di_ratio:.3f}</td>
                <td>{status_indicator}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html
