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
# Required Local Directories (minimal)
# ======================================================================

# Base path for Annex IV document generation
RISK_DIR = "docs/risk"
RELEASES_DIR = "docs/releases"
TEMPLATES_DIR = "docs/templates"
SAMPLE_INPUTS_PATH = f"{TEMPLATES_DIR}/sample_inputs.json"
RISK_REGISTER_PATH = f"{RISK_DIR}/risk_register.xlsx"

# Ensure minimal local directories exist
for dir_path in [RISK_DIR, RELEASES_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# ======================================================================
# Modal Configuration
# ======================================================================

MODAL_VOLUME_NAME = "credit-scoring"
MODAL_ENVIRONMENT = "main"

# Modal volume metadata and paths
VOLUME_METADATA = {
    "volume_name": MODAL_VOLUME_NAME,
    "secret_name": "credit-scoring-secrets",
    "app_name": "credit-scoring-app",
    "environment_name": MODAL_ENVIRONMENT,
    "model_path": "models/model.pkl",
    "preprocess_pipeline_path": "pipelines/preprocess_pipeline.pkl",
    "docs_dir": "docs",
    "risk_register_path": "docs/risk/risk_register.xlsx",
    "incident_log_path": "docs/risk/incident_log.json",
    "releases_dir": "docs/releases",
    "templates_dir": "docs/templates",
    "risk_dir": "docs/risk",
}


# ======================================================================
# Incident Reporting Configuration
# ======================================================================

# Slack webhook URL for incident reporting
SLACK_CHANNEL = "#credit-scoring-alerts"
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")


# ======================================================================
# Hazard Definitions
# ======================================================================

HAZARD_DEFINITIONS = {
    "bias_protected_groups": {
        "description": "Unfair bias against protected demographic groups",
        "trigger": lambda results, scores: (
            any(
                abs(v.get("selection_rate_disparity", 0)) > 0.2
                for v in results["fairness"]
                .get("fairness_metrics", {})
                .values()
                if isinstance(v, dict)
            )
        ),
        "severity": "high",
        "mitigation": (
            "Re-sample training data; add fairness constraints or post-processing techniques"
        ),
    },
    "low_accuracy": {
        "description": "Model accuracy below 0.75",
        "trigger": lambda results, scores: results["metrics"]["accuracy"]
        < 0.75,
        "severity": "medium",
        "mitigation": "Collect more data; tune hyper-parameters",
    },
    "data_quality": {
        "description": "Data-quality issues flagged during preprocessing",
        "trigger": lambda results, scores: (
            isinstance(results.get("data_quality"), dict)
            and results.get("data_quality", {}).get("issues_detected", False)
        ),
        "severity": "medium",
        "mitigation": "Tighten preprocessing / validation rules",
    },
    "model_complexity": {
        "description": "High model complexity reduces explainability",
        "trigger": lambda results, scores: (
            isinstance(results.get("model_info"), dict)
            and results.get("model_info", {}).get("complexity_score", 0) > 0.7
        ),
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
