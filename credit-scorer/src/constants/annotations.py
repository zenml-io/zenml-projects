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

"""Annotation constants."""

# Python ≥3.11 gives us StrEnum; fall back gracefully for 3.10
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum  # type: ignore

    class StrEnum(str, Enum):
        """Back-port of 3.11 StrEnum."""

        pass


class Pipelines(StrEnum):
    """Pipeline names used in ZenML."""

    FEATURE_ENGINEERING = "credit_scoring_feature_engineering"
    TRAINING = "credit_scoring_training"
    DEPLOYMENT = "credit_scoring_deployment"


class Artifacts(StrEnum):
    """ZenML artifact annotation names."""

    # ── model ──────────────────────────────────────────────────
    MODEL = "credit_scorer"

    # ── generic ────────────────────────────────────────────────
    CREDIT_SCORING_DF = "credit_scoring_df"

    # ── feature-engineering ────────────────────────────────────
    TRAIN_DATASET = "train_df"
    TEST_DATASET = "test_df"
    PREPROCESS_PIPELINE = "preprocess_pipeline"
    PREPROCESSING_METADATA = "preprocessing_metadata"
    WHYLOGS_PROFILE = "whylogs_profile"

    # ── training ───────────────────────────────────────────────
    OPTIMAL_THRESHOLD = "optimal_threshold"
    EVALUATION_RESULTS = "evaluation_results"
    EVAL_VISUALIZATION = "evaluation_visualization"
    RISK_SCORES = "risk_scores"
    FAIRNESS_REPORT = "fairness_report"
    RISK_REGISTER = "risk_register"

    # ── deployment / compliance ────────────────────────────────
    APPROVED = "approved"
    APPROVAL_RECORD = "approval_record"
    DEPLOYMENT_INFO = "deployment_info"
    MONITORING_PLAN = "monitoring_plan"
    INCIDENT_REPORT = "incident_report"
    COMPLIANCE_RECORD = "compliance_record"
    SBOM_ARTIFACT = "sbom_artifact"
    ANNEX_IV_PATH = "annex_iv_path"
    RUN_RELEASE_DIR = "run_release_dir"
    COMPLIANCE_DASHBOARD_HTML = "compliance_dashboard_html"
