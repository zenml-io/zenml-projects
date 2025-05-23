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

"""Risk constants."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict


class HazardSeverity(str, Enum):
    """Severity levels for model hazards."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Hazard:
    """Model hazard definition with trigger condition and mitigation strategy."""

    description: str
    trigger: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    severity: HazardSeverity
    mitigation: str

    def is_triggered(
        self, results: Dict[str, Any], scores: Dict[str, Any]
    ) -> bool:
        """Check if this hazard is triggered by the given results and scores."""
        return self.trigger(results, scores)


class Hazards:
    """Collection of model hazard definitions."""

    BIAS_PROTECTED_GROUPS = Hazard(
        description="Unfair bias against protected demographic groups",
        trigger=lambda results, scores: (
            any(
                abs(v.get("selection_rate_disparity", 0)) > 0.2
                for v in results["fairness"]
                .get("fairness_metrics", {})
                .values()
                if isinstance(v, dict)
            )
        ),
        severity=HazardSeverity.HIGH,
        mitigation=(
            "Re-sample training data; add fairness constraints or post-processing techniques"
        ),
    )

    LOW_ACCURACY = Hazard(
        description="Model accuracy below 0.75",
        trigger=lambda results, scores: results["metrics"]["accuracy"] < 0.75,
        severity=HazardSeverity.MEDIUM,
        mitigation="Collect more data; tune hyper-parameters",
    )

    DATA_QUALITY = Hazard(
        description="Data-quality issues flagged during preprocessing",
        trigger=lambda results, scores: (
            isinstance(results.get("data_quality"), dict)
            and results.get("data_quality", {}).get("issues_detected", False)
        ),
        severity=HazardSeverity.MEDIUM,
        mitigation="Tighten preprocessing / validation rules",
    )

    MODEL_COMPLEXITY = Hazard(
        description="High model complexity reduces explainability",
        trigger=lambda results, scores: (
            isinstance(results.get("model_info"), dict)
            and results.get("model_info", {}).get("complexity_score", 0) > 0.7
        ),
        severity=HazardSeverity.LOW,
        mitigation="Consider simpler model; add SHAP / LIME explanations",
    )

    DRIFT_VULNERABILITY = Hazard(
        description="ROC-AUC risk proxy > 0.3 indicates drift fragility",
        trigger=lambda results, scores: scores["risk_auc"] > 0.3,
        severity=HazardSeverity.MEDIUM,
        mitigation="Enable drift monitoring; schedule periodic retraining",
    )

    @classmethod
    def get_all(cls) -> Dict[str, Hazard]:
        """Get all hazards as a dictionary."""
        return {
            name: value
            for name, value in cls.__dict__.items()
            if isinstance(value, Hazard)
        }

    @classmethod
    def check_all(
        cls, results: Dict[str, Any], scores: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check all hazards against the provided results and scores."""
        return {
            name: hazard.is_triggered(results, scores)
            for name, hazard in cls.get_all().items()
        }
