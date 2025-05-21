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

from dataclasses import dataclass
from enum import Enum

from .compliance_constants import (
    EU_AI_ACT_ARTICLES,
)


class RiskStatus(Enum):
    """Enum representing risk status values."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class RiskCategory(Enum):
    """Enum representing risk category values."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EUArticle(Enum):
    """Enum representing EU AI Act articles."""

    ARTICLE_9 = "article_9"
    ARTICLE_10 = "article_10"
    ARTICLE_11 = "article_11"
    ARTICLE_12 = "article_12"
    ARTICLE_13 = "article_13"
    ARTICLE_14 = "article_14"
    ARTICLE_15 = "article_15"
    ARTICLE_16 = "article_16"
    ARTICLE_17 = "article_17"


@dataclass
class ComplianceThresholds:
    """Default thresholds for compliance calculations."""

    # Article 9 (Risk Management)
    risk_identification: float = 0.5  # Reduced from 1.0 for demo
    mitigation_coverage: float = 0.5  # Reduced from 0.7 for demo
    mitigation_completion: float = 0.3  # Reduced from 0.5 for demo
    risk_review_recency: int = 90  # Increased from 30 for demo
    overall_risk_score: float = (
        0.8  # Increased from 0.7 for demo (higher is tolerated)
    )

    # Article 10 (Data Governance)
    data_quality_score: float = 0.6  # Reduced from 0.8 for demo
    fairness_threshold: float = (
        0.3  # Increased from 0.2 for demo (higher is tolerated)
    )
    protected_attributes_coverage: float = 0.6  # Reduced from 0.8 for demo
    feature_importance_coverage: float = 0.6  # Reduced from 0.8 for demo

    # Article 11 (Technical Documentation)
    annex_iv_completeness: float = 0.6  # Reduced from 0.9 for demo
    sbom_completeness: float = 0.6  # Reduced from 0.95 for demo
    required_sections_coverage: float = 0.6  # Reduced from 0.9 for demo

    # Article 12 (Record Keeping)
    logging_completeness: float = 0.7  # Reduced from 0.95 for demo
    artifact_traceability: float = 0.6  # Reduced from 0.9 for demo
    audit_trail_completeness: float = 0.6  # Reduced from 0.9 for demo
    log_retention_days: int = 365 * 10  # 10 years
    # Default overall compliance threshold
    overall_compliance: float = 0.5  # Reduced from 0.8 for demo


# Risk register schema definition
RISK_REGISTER_SCHEMA = {
    "required_columns": [
        "Run_ID",
        "Timestamp",
        "Risk_overall",
        "Risk_auc",
        "Risk_bias",
        "Risk_category",
        "Status",
        "Risk_description",
        "Mitigation",
        "Article",
        "Mitigation_status",
        "Review_date",
    ],
    "optional_columns": [
        "Risk_ID",
        "Created_by",
        "Last_updated",
        "Comments",
    ],
    "column_types": {
        "Run_ID": str,
        "Timestamp": str,
        "Risk_overall": float,
        "Risk_auc": float,
        "Risk_bias": float,
        "Risk_category": str,
        "Status": str,
        "Risk_description": str,
        "Mitigation": str,
        "Article": str,
        "Mitigation_status": str,
        "Review_date": str,
    },
    "valid_values": {
        "Risk_category": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "Status": ["PENDING", "IN_PROGRESS", "COMPLETED"],
        "Mitigation_status": ["PENDING", "IN_PROGRESS", "COMPLETED"],
        "Article": [article for article in EU_AI_ACT_ARTICLES] + ["GENERAL"],
    },
}

# Dictionary of required sections in Annex IV documentation
ANNEX_IV_REQUIRED_SECTIONS = [
    "1. General Information",
    "2. System Description",
    "3. Technical Details",
    "4. Development Process",
    "5. Training Data",
    "6. Validation and Testing",
    "7. Performance Metrics",
    "8. Risk Management",
    "9. Human Oversight",
    "10. Accuracy and Robustness",
    "11. Compliance Documentation",
]
