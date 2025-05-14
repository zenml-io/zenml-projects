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
from pathlib import Path

# ======================================================================
# Dataset Configuration
# ======================================================================

CREDIT_SCORING_CSV_PATH = "data/credit_scoring.csv"
TARGET_COLUMN = "TARGET"

# List of potential sensitive attributes for fairness checks
SENSITIVE_ATTRIBUTES = [
    "CODE_GENDER",  # M or F
    "DAYS_BIRTH",  # age proxy (e.g. -9461 = 26 years old)
    "NAME_EDUCATION_TYPE",  # High school, Graduate, Academic degree
    "NAME_FAMILY_STATUS",  # Married, Single, Civil marriage, Separated, Divorced, Widow
    "NAME_HOUSING_TYPE",  # House / apartment, With parents, Rented apartment, Office apartment, Co-op apartment
]

# Ignore WhyLogs optional usage-telemetry API
os.environ["WHYLOGS_NO_ANALYTICS"] = "True"

# ======================================================================
# Pipeline Names
# ======================================================================
FEATURE_ENGINEERING_PIPELINE_NAME = "cs_feature_engineering"
TRAINING_PIPELINE_NAME = "cs_training"
DEPLOYMENT_PIPELINE_NAME = "cs_deployment"

# ======================================================================
# ZenML Artifact Names
# ======================================================================

# Feature engineering artifacts
MODEL_NAME = "credit_scoring_model"
TRAIN_DATASET_NAME = "cs_train_df"
TEST_DATASET_NAME = "cs_test_df"
PREPROCESS_PIPELINE_NAME = "cs_preprocess_pipeline"
PREPROCESS_METADATA_NAME = "cs_preprocessing_metadata"
DATA_PROFILE_NAME = "cs_data_profile"

# Training artifacts
EVALUATION_RESULTS_NAME = "cs_evaluation_results"
RISK_SCORES_NAME = "cs_risk_scores"
FAIRNESS_REPORT_NAME = "cs_fairness_report"
RISK_REGISTER_NAME = "cs_risk_register"

# Deployment artifacts
APPROVED_NAME = "cs_approved"
APPROVAL_RECORD_NAME = "cs_approval_record"
DEPLOYMENT_INFO_NAME = "cs_deployment_info"
MONITORING_PLAN_NAME = "cs_monitoring_plan"
INCIDENT_REPORT_NAME = "cs_incident_report"
COMPLIANCE_METADATA_NAME = "cs_compliance_metadata"
COMPLIANCE_REPORT_NAME = "cs_compliance_report"
MODEL_CARD_NAME = "cs_model_card"

# ======================================================================
# Modal Configuration
# ======================================================================

# Modal deployment settings
MODAL_VOLUME_NAME = "credit-scoring"
MODAL_APP_NAME = "credit-scoring-app"
MODAL_SECRET_NAME = "credit-scoring-secrets"
MODAL_ENVIRONMENT = "main"

# Maximum number of model versions to keep in the Modal Volume
MAX_MODEL_VERSIONS = 5

# Modal volume paths (for deployment)
MODAL_MODELS_DIR = "/models"
MODAL_PIPELINES_DIR = "/pipelines"
MODAL_COMPLIANCE_DIR = "/compliance"
MODAL_EVAL_RESULTS_DIR = "/evaluation"
MODAL_APPROVALS_DIR = f"{MODAL_COMPLIANCE_DIR}/approvals"
MODAL_MONITORING_DIR = f"{MODAL_COMPLIANCE_DIR}/monitoring"
MODAL_DEPLOYMENTS_DIR = f"{MODAL_COMPLIANCE_DIR}/deployments"
MODAL_FAIRNESS_DIR = f"{MODAL_COMPLIANCE_DIR}/fairness"
MODAL_REPORTS_DIR = f"{MODAL_COMPLIANCE_DIR}/reports"
MODAL_MANUAL_FILLS_DIR = f"{MODAL_COMPLIANCE_DIR}/manual_fills"
MODAL_RISK_DIR = f"{MODAL_COMPLIANCE_DIR}/risk"

# Default Modal artifact paths
MODAL_MODEL_PATH = f"{MODAL_MODELS_DIR}/model.pkl"
MODAL_PREPROCESS_PIPELINE_PATH = f"{MODAL_PIPELINES_DIR}/preprocess_pipeline.pkl"
MODAL_RISK_REGISTER_PATH = f"{MODAL_RISK_DIR}/risk_register.xlsx"
MODAL_INCIDENT_LOG_PATH = f"{MODAL_COMPLIANCE_DIR}/incident_log.json"

# Standard keys for volume_metadata dictionary
VOLUME_METADATA_KEYS = {
    "volume_name": MODAL_VOLUME_NAME,
    "environment_name": MODAL_ENVIRONMENT,
    "model_path": MODAL_MODEL_PATH,
    "preprocess_pipeline_path": MODAL_PREPROCESS_PIPELINE_PATH,
    "risk_register_path": MODAL_RISK_REGISTER_PATH,
    "incident_log_path": MODAL_INCIDENT_LOG_PATH,
    "compliance_dir": MODAL_COMPLIANCE_DIR,
    "fairness_dir": MODAL_FAIRNESS_DIR,
    "reports_dir": MODAL_REPORTS_DIR,
    "deployments_records_dir": MODAL_DEPLOYMENTS_DIR,
    "approvals_dir": MODAL_APPROVALS_DIR,
    "monitoring_dir": MODAL_MONITORING_DIR,
    "manual_fills_dir": MODAL_MANUAL_FILLS_DIR,
}

# ======================================================================
# Required Local Directories (minimal)
# ======================================================================

# Base path for Annex IV document generation
COMPLIANCE_DIR = Path(os.getcwd()) / "compliance"
REPORTS_DIR = COMPLIANCE_DIR / "reports"
MANUAL_FILLS_DIR = COMPLIANCE_DIR / "manual_fills"
TEMPLATES_DIR = COMPLIANCE_DIR / "templates"

# Ensure minimal local directories exist
for dir_path in [REPORTS_DIR, MANUAL_FILLS_DIR, TEMPLATES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Hazard Definitions
# ======================================================================

HAZARD_DEFINITIONS = {
    "bias_protected_groups": {
        "description": "Unfair bias against protected demographic groups",
        "trigger": lambda results, scores: (
            any(
                abs(v.get("selection_rate_disparity", 0)) > 0.2
                for v in results["fairness"].values()
            )
        ),
        "severity": "high",
        "mitigation": (
            "Re-sample training data; add fairness constraints or post-processing techniques"
        ),
    },
    "low_accuracy": {
        "description": "Model accuracy below 0.75",
        "trigger": lambda results, scores: results["metrics"]["accuracy"] < 0.75,
        "severity": "medium",
        "mitigation": "Collect more data; tune hyper-parameters",
    },
    "data_quality": {
        "description": "Data-quality issues flagged during preprocessing",
        "trigger": lambda results, scores: results.get("data_quality", {}).get(
            "issues_detected", False
        ),
        "severity": "medium",
        "mitigation": "Tighten preprocessing / validation rules",
    },
    "model_complexity": {
        "description": "High model complexity reduces explainability",
        "trigger": lambda results, scores: results.get("model_info", {}).get("complexity_score", 0)
        > 0.7,
        "severity": "low",
        "mitigation": "Consider simpler model; add SHAP / LIME explanations",
    },
    "drift_vulnerability": {
        "description": "ROC-AUC risk proxy > 0.3 indicates drift fragility",
        "trigger": lambda results, scores: scores["risk_auc"] > 0.3,
        "severity": "medium",
        "mitigation": "Enable drift monitoring; schedule periodic retraining",
    },
}


# ======================================================================
# Model Approval Thresholds
# ======================================================================

APPROVAL_THRESHOLDS = {
    "accuracy": 0.80,  # Minimum acceptable accuracy
    "bias_disparity": 0.20,  # Maximum acceptable disparity
    "risk_score": 0.40,  # Maximum acceptable overall risk
}

# ======================================================================
# Incident Reporting Configuration
# ======================================================================

# Slack webhook URL for incident reporting
SLACK_CHANNEL = "#credit-scoring-alerts"
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
