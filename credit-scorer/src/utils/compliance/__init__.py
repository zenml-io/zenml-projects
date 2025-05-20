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

from typing import Any, Dict, List, Optional

from .data_loader import ComplianceDataLoader
from .exceptions import (
    ComplianceCalculationError,
    ComplianceDataError,
    ComplianceError,
)
from .orchestrator import ComplianceOrchestrator
from .schemas import ComplianceThresholds, EUArticle, RiskCategory, RiskStatus


# Main public interface
def calculate_compliance(
    release_id: Optional[str] = None,
    risk_register_path: Optional[str] = None,
    articles: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate full EU AI Act compliance for the specified system.

    This is the main entry point for compliance calculations. It loads necessary data,
    calculates compliance metrics for all relevant articles, and returns comprehensive results.

    Args:
        release_id: Optional specific release to analyze.
                  If not provided, the latest release will be used.
        risk_register_path: Optional path to risk register.
                          If not provided, the default path will be used.
        articles: Optional list of articles to calculate compliance for.
                 If not provided, all available calculators will be used.

    Returns:
        Complete compliance results containing:
        - overall: Overall compliance metrics
        - articles: Per-article compliance details
        - warnings: Any warnings encountered during calculation
        - release_id: The release ID used for calculations
        - preprocessed_data: Additional metrics from data preprocessing

    Raises:
        ComplianceError: If compliance calculation fails.

    Example:
        ```python
        from src.utils.compliance import calculate_compliance

        # Calculate compliance for all articles using latest release
        results = calculate_compliance()
        print(f"Overall compliance: {results['overall']['overall_compliance_score']:.1f}%")

        # Calculate compliance for specific articles and release
        results = calculate_compliance(
            release_id="a819ed29-5a68-4bbb-94a3-5aad9f67a773",
            articles=["article_9", "article_10"]
        )
        ```
    """
    orchestrator = ComplianceOrchestrator()
    return orchestrator.calculate_full_compliance(
        release_id, risk_register_path, articles
    )


# Expose individual classes for advanced use
__all__ = [
    "calculate_compliance",
    "ComplianceOrchestrator",
    "ComplianceDataLoader",
    "ComplianceError",
    "ComplianceDataError",
    "ComplianceCalculationError",
    "ComplianceThresholds",
    "EUArticle",
    "RiskStatus",
    "RiskCategory",
]
