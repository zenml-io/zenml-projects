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

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..exceptions import ComplianceCalculationError
from ..schemas import ComplianceThresholds


class BaseComplianceCalculator(ABC):
    """Abstract base class for article compliance calculators."""

    def __init__(self, thresholds: Optional[ComplianceThresholds] = None):
        """Initialize the calculator with optional custom thresholds.

        Args:
            thresholds: Optional custom thresholds to use for compliance calculations.
                      If not provided, defaults will be used.
        """
        self.thresholds = thresholds or ComplianceThresholds()

    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, Any]:
        """Calculate compliance metrics for the article.

        Args:
            **kwargs: Keyword arguments with data required for calculations.
                    Each calculator implementation will define its own required arguments.

        Returns:
            Dict with keys:
            - compliance_score: Overall compliance score (0-100)
            - metrics: Detailed metrics used for calculation
            - findings: Detailed findings and recommendations

        Raises:
            ComplianceCalculationError: If calculation fails.
        """
        pass

    def _create_finding(
        self, finding_type: str, message: str, recommendation: str
    ) -> Dict[str, str]:
        """Helper to create standardized findings.

        Args:
            finding_type: Type of finding (e.g., "critical", "warning", "info", "positive")
            message: Descriptive message about the finding
            recommendation: Recommendation for addressing the finding

        Returns:
            Dictionary with standardized finding format
        """
        return {
            "type": finding_type,
            "message": message,
            "recommendation": recommendation,
        }

    def _calculate_weighted_score(
        self, metrics: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """Calculate weighted average of metrics.

        Args:
            metrics: Dictionary mapping metric names to scores
            weights: Dictionary mapping metric names to their weights

        Returns:
            Weighted average score

        Raises:
            ComplianceCalculationError: If calculation fails due to missing metrics.
        """
        if not all(key in metrics for key in weights):
            missing_metrics = [key for key in weights if key not in metrics]
            raise ComplianceCalculationError(
                f"Cannot calculate weighted score. Missing metrics: {missing_metrics}"
            )

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(metrics[key] * weights[key] for key in weights)
        return weighted_sum / total_weight

    def _format_percentage(self, value: float) -> str:
        """Format a float as a percentage string with one decimal place.

        Args:
            value: Float value between 0 and 1

        Returns:
            Formatted percentage string
        """
        return f"{value * 100:.1f}%"

    def _add_overall_finding(
        self,
        findings: List[Dict[str, str]],
        compliance_score: float,
        article_name: str,
    ) -> None:
        """Add an overall finding based on the compliance score.

        Args:
            findings: List of findings to append to
            compliance_score: Overall compliance score (0-100)
            article_name: Name of the article for context
        """
        if compliance_score >= 85:
            findings.append(
                self._create_finding(
                    "positive",
                    f"{article_name} compliance is high ({compliance_score:.1f}%).",
                    f"Continue maintaining compliance with {article_name}.",
                )
            )
        elif compliance_score >= 70:
            findings.append(
                self._create_finding(
                    "info",
                    f"{article_name} compliance is moderate ({compliance_score:.1f}%).",
                    f"Address the identified findings to improve {article_name} compliance.",
                )
            )
        else:
            findings.append(
                self._create_finding(
                    "critical",
                    f"{article_name} compliance is low ({compliance_score:.1f}%).",
                    f"Urgently address the identified findings to improve {article_name} compliance.",
                )
            )
