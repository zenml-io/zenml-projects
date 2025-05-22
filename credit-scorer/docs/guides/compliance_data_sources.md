## Compliance Data Sources Overview

This table describes the data sources used for tracking compliance with the EU AI Act articles. These data sources are used to calculate compliance metrics, which are then used to generate compliance reports.

| Data Source              | Description                                                        | File Path                                              | Format    |
| ------------------------ | ------------------------------------------------------------------ | ------------------------------------------------------ | --------- |
| Risk Register            | Contains identified risks, their severity, and mitigation measures | `docs/risk/risk_register.xlsx`                         | Excel     |
| Incident Log             | Records of compliance incidents and their resolution status        | `docs/risk/incident_log.json`                          | JSON      |
| Release Artifacts        | Generated documentation for each model release                     | `docs/releases/{release_id}/`                          | Directory |
| Evaluation Results       | Metrics from model evaluation, including performance and fairness  | `docs/releases/{release_id}/evaluation_results.yaml`   | YAML      |
| Risk Scores              | Risk assessment scores for the model                               | `docs/releases/{release_id}/risk_scores.yaml`          | YAML      |
| Monitoring Plan          | Post-market monitoring configuration                               | `docs/releases/{release_id}/monitoring_plan.json`      | JSON      |
| Annex IV Documentation   | Technical documentation required by EU AI Act                      | `docs/releases/{release_id}/annex_iv.md`               | Markdown  |
| SBOM                     | Software Bill of Materials                                         | `docs/releases/{release_id}/sbom.json`                 | JSON      |
| Approval Record          | Human approval documentation for deployment                        | `docs/releases/{release_id}/approval_record.json`      | JSON      |
| Git Information          | Git commit history and version control metadata                    | `docs/releases/{release_id}/git_info.md`               | Markdown  |
| Log Metadata             | Metadata about pipeline execution logs                             | `docs/releases/{release_id}/log_metadata.json`         | JSON      |
| Model Card               | Model specifications, use cases, and limitations                   | `docs/releases/{release_id}/model_card.md`             | Markdown  |
| Evaluation Visualization | Interactive visualization of model evaluation results              | `docs/releases/{release_id}/eval_visualization.html`   | HTML      |
| WhyLogs Profile          | Data profiling report with quality metrics                         | `docs/releases/{release_id}/whylogs_profile.html`      | HTML      |
| Compliance Dashboard     | Visual summary of compliance status and metrics                    | `docs/releases/{release_id}/compliance_dashboard.html` | HTML      |
| Release Summary          | Overview and index of release artifacts                            | `docs/releases/{release_id}/README.md`                 | Markdown  |
