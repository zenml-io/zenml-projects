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


# EU AI Act Articles relevant to high-risk AI systems (like credit scoring)
EU_AI_ACT_ARTICLES = [
    "article_9",
    "article_10",
    "article_11",
    "article_12",
    "article_13",
    "article_14",
    "article_15",
    "article_16",
    "article_17",
]

# Article titles and descriptions
ARTICLE_DESCRIPTIONS = {
    "article_9": {
        "title": "Risk Management System",
        "description": "Establish and maintain a risk management system throughout the entire lifecycle of the high-risk AI system",
        "requirements": [
            "Identify and analyze known and foreseeable risks",
            "Estimate and evaluate risks that may emerge",
            "Adopt risk management measures",
            "Test and validate the efficacy of risk mitigation measures",
        ],
    },
    "article_10": {
        "title": "Data and Data Governance",
        "description": "Implement data governance and management practices regarding training, validation, and testing datasets",
        "requirements": [
            "Establish data governance practices",
            "Examine datasets for biases",
            "Identify data anomalies or shortcomings",
            "Ensure data quality, relevance, and representativeness",
        ],
    },
    "article_11": {
        "title": "Technical Documentation",
        "description": "Comprehensive technical documentation demonstrating compliance",
        "requirements": [
            "Document system description",
            "Detail design specifications",
            "Record development processes",
            "Maintain version control documentation",
        ],
    },
    "article_12": {
        "title": "Record Keeping",
        "description": "Automatic recording of events and maintaining logs throughout system lifecycle",
        "requirements": [
            "Implement logging mechanisms for key system events",
            "Maintain audit trails for system actions and decisions",
            "Ensure traceability and reproducibility",
            "Define data retention policies",
        ],
    },
    "article_13": {
        "title": "Transparency and Provision of Information",
        "description": "Provide transparent information about AI system operation and monitoring",
        "requirements": [
            "Document system capabilities and limitations",
            "Explain data governance measures",
            "Provide user instructions",
            "Disclose monitoring mechanisms",
        ],
    },
    "article_14": {
        "title": "Human Oversight",
        "description": "Enable effective human oversight through appropriate measures",
        "requirements": [
            "Implement human oversight mechanisms",
            "Enable human intervention capabilities",
            "Document oversight procedures",
            "Ensure adequate training for human overseers",
        ],
    },
    "article_15": {
        "title": "Accuracy, Robustness and Cybersecurity",
        "description": "Develop systems to achieve appropriate levels of accuracy, robustness, and security",
        "requirements": [
            "Define accuracy metrics and thresholds",
            "Implement robustness testing",
            "Establish cyber security controls",
            "Document resilience measures",
        ],
    },
    "article_16": {
        "title": "Quality Management System",
        "description": "Implement a quality management system to ensure compliance",
        "requirements": [
            "Establish quality management procedures",
            "Document compliance verification methods",
            "Implement post-market monitoring system",
            "Define responsibilities and resources",
        ],
    },
    "article_17": {
        "title": "Post-market Monitoring",
        "description": "Establish a post-market monitoring system to detect and address issues",
        "requirements": [
            "Implement data collection systems",
            "Define monitoring frequency",
            "Establish incident reporting mechanisms",
            "Document remediation procedures",
        ],
    },
}

# Mapping from Articles to data sources and metrics
COMPLIANCE_DATA_SOURCES = {
    "article_9": {
        "primary_sources": ["risk_register", "risk_scores"],
        "metrics": {
            "identified_risks": {
                "source": "risk_register",
                "field": "risks_count",
                "threshold": 0,  # Any identified risks should be documented
                "description": "Number of identified risks in the risk register",
            },
            "mitigated_risks": {
                "source": "risk_register",
                "field": "mitigated_risks_percentage",
                "threshold": 0.7,  # At least 70% of risks should have mitigation
                "description": "Percentage of risks with defined mitigation measures",
            },
            "risk_score": {
                "source": "risk_scores",
                "field": "overall",
                "threshold": 0.7,  # Risk score should be below 0.7 (lower is better)
                "description": "Overall risk score from assessment (lower is better)",
            },
        },
    },
    "article_10": {
        "primary_sources": ["evaluation_results", "preprocessing_metadata"],
        "metrics": {
            "data_quality": {
                "source": "evaluation_results",
                "field": "data_quality.score",
                "threshold": 0.8,  # Data quality score should be at least 80%
                "description": "Data quality score from data profiling",
            },
            "data_representation": {
                "source": "evaluation_results",
                "field": "data_quality.representation_score",
                "threshold": 0.7,  # Representation score should be at least 70%
                "description": "Score measuring representativeness of data",
            },
            "feature_relevance": {
                "source": "preprocessing_metadata",
                "field": "feature_importance_coverage",
                "threshold": 0.8,  # Feature importance coverage should be at least 80%
                "description": "Coverage of features with documented importance",
            },
        },
    },
    "article_11": {
        "primary_sources": ["annex_iv", "sbom"],
        "metrics": {
            "documentation_completeness": {
                "source": "annex_iv",
                "field": "completeness_score",
                "threshold": 0.9,  # Documentation should be at least 90% complete
                "description": "Completeness score of technical documentation",
            },
            "version_control": {
                "source": "sbom",
                "field": "completeness",
                "threshold": 0.95,  # SBOM should be at least 95% complete
                "description": "Completeness of software bill of materials",
            },
        },
    },
    "article_12": {
        "primary_sources": ["run_history", "pipeline_logs"],
        "metrics": {
            "logging_completeness": {
                "source": "pipeline_logs",
                "field": "completion_percentage",
                "threshold": 0.95,  # Logging should be at least 95% complete
                "description": "Completeness of system event logging",
            },
            "artifact_traceability": {
                "source": "run_history",
                "field": "artifact_lineage_completeness",
                "threshold": 0.9,  # Artifact lineage should be at least 90% complete
                "description": "Completeness of artifact lineage tracing",
            },
            "audit_trail": {
                "source": "run_history",
                "field": "audit_trail_completeness",
                "threshold": 0.9,  # Audit trail should be at least 90% complete
                "description": "Completeness of system audit trail",
            },
        },
    },
    "article_13": {
        "primary_sources": ["annex_iv", "model_card"],
        "metrics": {
            "transparency_score": {
                "source": "annex_iv",
                "field": "transparency_sections_covered",
                "threshold": 0.8,  # Transparency documentation should be at least 80% complete
                "description": "Coverage of transparency requirements in documentation",
            },
            "user_information": {
                "source": "model_card",
                "field": "completeness",
                "threshold": 0.9,  # Model card should be at least 90% complete
                "description": "Completeness of user-facing model information",
            },
        },
    },
    "article_14": {
        "primary_sources": ["approval_records", "monitoring_plan"],
        "metrics": {
            "oversight_implementation": {
                "source": "approval_records",
                "field": "human_approval_count",
                "threshold": 1,  # At least one human approval must exist
                "description": "Number of human approval records",
            },
            "intervention_capabilities": {
                "source": "monitoring_plan",
                "field": "human_intervention_procedures_count",
                "threshold": 2,  # At least two intervention procedures should be defined
                "description": "Number of defined human intervention procedures",
            },
        },
    },
    "article_15": {
        "primary_sources": ["evaluation_results", "risk_scores"],
        "metrics": {
            "accuracy": {
                "source": "evaluation_results",
                "field": "metrics.accuracy",
                "threshold": 0.8,  # Accuracy should be at least 80%
                "description": "Model accuracy on test data",
            },
            "robustness": {
                "source": "evaluation_results",
                "field": "robustness_score",
                "threshold": 0.7,  # Robustness score should be at least 70%
                "description": "Score measuring model robustness",
            },
            "fairness": {
                "source": "evaluation_results",
                "field": "fairness.bias_flag",
                "threshold": False,  # No bias flag should be raised
                "description": "Flag indicating presence of bias in model",
            },
        },
    },
    "article_16": {
        "primary_sources": ["annex_iv", "sbom", "monitoring_plan"],
        "metrics": {
            "qms_procedures": {
                "source": "annex_iv",
                "field": "qms_procedures_count",
                "threshold": 4,  # At least 4 QMS procedures should be defined
                "description": "Number of defined quality management procedures",
            },
            "documentation_quality": {
                "source": "annex_iv",
                "field": "documentation_quality_score",
                "threshold": 0.85,  # Documentation quality should be at least 85%
                "description": "Quality score for compliance documentation",
            },
        },
    },
    "article_17": {
        "primary_sources": ["monitoring_plan", "incident_log"],
        "metrics": {
            "monitoring_coverage": {
                "source": "monitoring_plan",
                "field": "monitoring_parameters_count",
                "threshold": 3,  # At least 3 parameters should be monitored
                "description": "Number of parameters being monitored",
            },
            "incident_response": {
                "source": "monitoring_plan",
                "field": "response_procedures_count",
                "threshold": 2,  # At least 2 response procedures should be defined
                "description": "Number of defined incident response procedures",
            },
            "incident_resolution": {
                "source": "incident_log",
                "field": "resolved_incidents_percentage",
                "threshold": 0.8,  # At least 80% of incidents should be resolved
                "description": "Percentage of resolved incidents",
            },
        },
    },
}

# Default file paths for compliance data sources
DEFAULT_COMPLIANCE_PATHS = {
    "risk_register": "docs/risk/risk_register.xlsx",
    "incident_log": "docs/risk/incident_log.json",
    "releases_dir": "docs/releases",
    "evaluation_results": "evaluation_results.yaml",
    "risk_scores": "risk_scores.yaml",
    "monitoring_plan": "monitoring_plan.json",
    "annex_iv": "annex_iv.md",
    "sbom": "sbom.json",
    "model_card": "model_card.md",
    "config_file": "compliance_articles.yaml",
}
