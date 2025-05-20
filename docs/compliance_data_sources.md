# Compliance Data Sources

This document describes the data sources used for tracking compliance with the EU AI Act articles. These data sources are used to calculate compliance metrics, which are then used to generate compliance reports.

## Data Sources Overview

| Data Source            | Description                                                        | File Path                                              | Format           |
| ---------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ | ---------------- |
| Risk Register          | Contains identified risks, their severity, and mitigation measures | `docs/risk/risk_register.xlsx`                         | Excel            |
| Incident Log           | Records of compliance incidents and their resolution status        | `docs/risk/incident_log.json`                          | JSON             |
| Release Artifacts      | Generated documentation for each model release                     | `docs/releases/{release_id}/`                          | Directory        |
| Evaluation Results     | Metrics from model evaluation, including performance and fairness  | `docs/releases/{release_id}/evaluation_results.yaml`   | YAML             |
| Risk Scores            | Risk assessment scores for the model                               | `docs/releases/{release_id}/risk_scores.yaml`          | YAML             |
| Monitoring Plan        | Post-market monitoring configuration                               | `docs/releases/{release_id}/monitoring_plan.json`      | JSON             |
| Annex IV Documentation | Technical documentation required by EU AI Act                      | `docs/releases/{release_id}/annex_iv_cs_deployment.md` | Markdown         |
| SBOM                   | Software Bill of Materials                                         | `docs/releases/{release_id}/sbom.json`                 | JSON             |
| Run History            | Pipeline execution history from ZenML                              | ZenML database                                         | Database records |
| Pipeline Logs          | Detailed logs from pipeline execution                              | ZenML logs                                             | Log files        |

## Data Sources by Article

### Article 9: Risk Management System

**Primary Data Sources:**

- **Risk Register** (`docs/risk/risk_register.xlsx`)
  - Contains risk identification, analysis, and mitigation information
  - Includes risk severity, likelihood, impact, and mitigation measures
- **Risk Scores** (`docs/releases/{release_id}/risk_scores.yaml`)
  - Contains risk assessment scores for various risk categories
  - Includes overall risk score and individual risk component scores

**Metrics Extracted:**

- Number of identified risks
- Percentage of risks with defined mitigation measures
- Overall risk score (lower is better)

### Article 10: Data and Data Governance

**Primary Data Sources:**

- **Evaluation Results** (`docs/releases/{release_id}/evaluation_results.yaml`)
  - Contains data quality metrics from WhyLogs profiles
  - Includes data representation scores across protected attributes
- **Preprocessing Metadata** (from ZenML artifacts)
  - Contains information about data transformations
  - Includes feature importance and relevance metrics

**Metrics Extracted:**

- Data quality score
- Data representation score
- Feature importance coverage

### Article 11: Technical Documentation

**Primary Data Sources:**

- **Annex IV Documentation** (`docs/releases/{release_id}/annex_iv_cs_deployment.md`)
  - Comprehensive technical documentation required by EU AI Act
  - Includes system description, design specifications, and version control
- **SBOM** (`docs/releases/{release_id}/sbom.json`)
  - Software Bill of Materials
  - Includes dependencies, versions, and licenses

**Metrics Extracted:**

- Documentation completeness score
- Version control completeness

### Article 12: Record Keeping

**Primary Data Sources:**

- **Run History** (ZenML database)
  - Contains execution history of all pipeline runs
  - Tracks artifact lineage and provenance
  - Provides traceability for all system decisions
- **Pipeline Logs** (ZenML logs)
  - Detailed logs of all pipeline steps
  - Event-level logging of system actions

**Metrics Extracted:**

- Logging completeness percentage
- Artifact lineage completeness
- Audit trail completeness

### Article 13: Transparency and Provision of Information

**Primary Data Sources:**

- **Annex IV Documentation** (`docs/releases/{release_id}/annex_iv_cs_deployment.md`)
  - Contains transparency sections for users
  - Includes system capabilities and limitations
- **Model Card** (generated during deployment)
  - User-facing information about model behavior
  - Includes intended use, performance metrics, and limitations

**Metrics Extracted:**

- Transparency documentation coverage
- Model card completeness

### Article 14: Human Oversight

**Primary Data Sources:**

- **Approval Records** (from ZenML artifacts)
  - Records of human approvals for model deployment
  - Includes approval timestamps, approver, and decision
- **Monitoring Plan** (`docs/releases/{release_id}/monitoring_plan.json`)
  - Contains human intervention procedures
  - Defines escalation paths and responsible parties

**Metrics Extracted:**

- Number of human approval records
- Number of defined human intervention procedures

### Article 15: Accuracy, Robustness and Cybersecurity

**Primary Data Sources:**

- **Evaluation Results** (`docs/releases/{release_id}/evaluation_results.yaml`)
  - Contains accuracy and roc_auc
  - Includes fairness metrics and bias detection results
- **Risk Scores** (`docs/releases/{release_id}/risk_scores.yaml`)
  - Contains robustness assessment

**Metrics Extracted:**

- Model accuracy on test data
- Robustness score
- Bias flag (indicating presence of bias)

### Article 16: Quality Management System

**Primary Data Sources:**

- **Annex IV Documentation** (`docs/releases/{release_id}/annex_iv_cs_deployment.md`)
  - Contains quality management procedures
  - Documents compliance verification methods
- **SBOM** (`docs/releases/{release_id}/sbom.json`)
  - Component and dependency documentation
- **Monitoring Plan** (`docs/releases/{release_id}/monitoring_plan.json`)
  - Post-market monitoring configuration

**Metrics Extracted:**

- Number of defined quality management procedures
- Documentation quality score

### Article 17: Post-market Monitoring

**Primary Data Sources:**

- **Monitoring Plan** (`docs/releases/{release_id}/monitoring_plan.json`)
  - Defines monitoring parameters, frequency, and thresholds
  - Includes response procedures and responsible parties
- **Incident Log** (`docs/risk/incident_log.json`)
  - Records incidents discovered through monitoring
  - Tracks resolution status of incidents

**Metrics Extracted:**

- Number of parameters being monitored
- Number of defined response procedures
- Percentage of resolved incidents

## How to Use These Data Sources

The data sources listed above are used to calculate compliance metrics defined in `src/utils/compliance_constants.py`. The compliance metrics are calculated during various pipeline steps and stored as ZenML artifacts. These metrics can be used to:

1. Generate compliance reports
2. Monitor compliance status
3. Identify areas for improvement
4. Demonstrate compliance to auditors or regulators

The compliance metrics are calculated using the `calculate_compliance_metrics` function in the compliance utility module.
