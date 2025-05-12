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

HF_DATASET_NAME = "spectrallabs/credit-scoring-training-dataset"
HF_DATASET_FILE = "0xFDC1BE05aD924e6Fc4Ab2c6443279fF7C0AB5544_training_data.parquet"
TARGET_COLUMN = "target"

# Ignore WhyLogs optional usage-telemetry API
os.environ["WHYLOGS_NO_ANALYTICS"] = "True"

# Feature engineering pipeline
FEATURE_ENGINEERING_PIPELINE_NAME = "credit_scoring_feature_engineering"
TRAIN_DATASET_NAME = "credit_scoring_train_df"
TEST_DATASET_NAME = "credit_scoring_test_df"
PREPROCESS_PIPELINE_NAME = "credit_scoring_preprocess_pipeline"
PREPROCESS_METADATA_NAME = "credit_scoring_preprocessing_metadata"

# Training pipeline
TRAINING_PIPELINE_NAME = "credit_scoring_training"
EVALUATION_RESULTS_NAME = "credit_scoring_evaluation_results"
RISK_SCORES_NAME = "credit_scoring_risk_scores"

# Deployment pipeline
DEPLOYMENT_PIPELINE_NAME = "credit_scoring_deployment"
APPROVED_NAME = "credit_scoring_approved"
DEPLOYMENT_INFO_NAME = "credit_scoring_deployment_info"
MONITORING_PLAN_NAME = "credit_scoring_monitoring_plan"
INCIDENT_REPORT_NAME = "credit_scoring_incident_report"

# Modal deployment
MODAL_VOLUME_NAME = "credit-scoring"
MODAL_DEPLOYMENT_NAME = "credit_scoring_modal_deployment"
MODAL_SECRET_NAME = "credit-scoring-secrets"

# Compliance artifacts
COMPLIANCE_METADATA_NAME = "credit_scoring_compliance_metadata"
COMPLIANCE_REPORT_NAME = "credit_scoring_compliance_report"
MODEL_CARD_NAME = "credit_scoring_model_card"
