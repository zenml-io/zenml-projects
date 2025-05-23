import logging
import os
from typing import Any, Dict, List, Optional

from .compliance_calculator import ComplianceCalculator
from .compliance_constants import (
    DEFAULT_COMPLIANCE_PATHS,
    EU_AI_ACT_ARTICLES,
)
from .data_loader import ComplianceDataLoader
from .exceptions import ComplianceError

logger = logging.getLogger(__name__)


class ComplianceOrchestrator:
    """Main coordinator for config-driven compliance calculations."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with data loader and config path.

        Args:
            config_path: Path to compliance_articles.yaml file
        """
        self.data_loader = ComplianceDataLoader()

        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(
                base_dir, DEFAULT_COMPLIANCE_PATHS["config_file"]
            )
        else:
            self.config_path = config_path

    def calculate_full_compliance(
        self,
        release_id: Optional[str] = None,
        risk_register_path: Optional[str] = None,
        articles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Calculate compliance for specified articles using YAML config.

        Args:
            release_id: Specific release to analyze (uses latest if None)
            risk_register_path: Path to risk register (uses default if None)
            articles: Articles to check (uses all if None)

        Returns:
            Dict with overall and per-article compliance results
        """
        try:
            # Load common data
            data = self._load_common_data(release_id, risk_register_path)

            # Determine articles to process
            articles_to_process = articles or EU_AI_ACT_ARTICLES

            # Calculate compliance per article
            article_results = {}
            for article_id in articles_to_process:
                try:
                    calculator = ComplianceCalculator(
                        self.config_path, article_id
                    )
                    article_results[article_id] = calculator.calculate(**data)
                except Exception as e:
                    logger.error(f"Failed to calculate {article_id}: {e}")
                    article_results[article_id] = {
                        "compliance_score": 0,
                        "metrics": {},
                        "findings": [
                            {
                                "type": "critical",
                                "message": f"Calculation failed: {e}",
                                "recommendation": "Check configuration and data sources",
                            }
                        ],
                    }

            # Calculate overall compliance
            overall_result = self._calculate_overall_compliance(
                article_results
            )

            return {
                "overall": overall_result,
                "articles": article_results,
                "warnings": data.get("warnings", []),
                "release_id": data.get("release_id"),
            }

        except Exception as e:
            logger.error(f"Compliance calculation failed: {e}")
            raise ComplianceError(f"Failed to calculate compliance: {e}")

    def _load_common_data(
        self, release_id: Optional[str], risk_register_path: Optional[str]
    ) -> Dict[str, Any]:
        """Load all common data sources needed for compliance calculations."""
        warnings = []

        # Load risk register
        risk_df, risk_warnings = self.data_loader.load_risk_register(
            risk_register_path
        )
        warnings.extend(risk_warnings)

        # Get release ID
        if release_id is None:
            release_id = self.data_loader.get_latest_release_id()
            if release_id:
                logger.info(f"Using latest release: {release_id}")

        # Load evaluation results
        eval_results = {}
        if release_id:
            try:
                eval_results, eval_warnings = (
                    self.data_loader.load_evaluation_results(release_id)
                )
                warnings.extend(eval_warnings)
            except Exception as e:
                logger.warning(f"Could not load evaluation results: {e}")
                warnings.append(f"Could not load evaluation results: {e}")

        # Load risk scores - ADD THIS SECTION
        risk_scores = {}
        if release_id:
            risk_scores_path = os.path.join(
                DEFAULT_COMPLIANCE_PATHS["releases_dir"],
                release_id,
                "risk_scores.yaml",
            )
            if os.path.exists(risk_scores_path):
                try:
                    import yaml

                    with open(risk_scores_path, "r") as f:
                        risk_scores = yaml.safe_load(f)
                    logger.info(f"Loaded risk scores from {risk_scores_path}")
                except Exception as e:
                    logger.warning(f"Failed to load risk scores: {e}")
                    warnings.append(f"Failed to load risk scores: {e}")

        # Load incident log
        incident_log = []
        try:
            incident_log, incident_warnings = (
                self.data_loader.load_incident_log()
            )
            warnings.extend(incident_warnings)
        except Exception as e:
            logger.warning(f"Could not load incident log: {e}")
            warnings.append(f"Could not load incident log: {e}")

        # Preprocess risk data
        preprocessed_data = self.data_loader.preprocess_compliance_data(
            risk_df
        )

        return {
            "risk_register": risk_df,
            "evaluation_results": eval_results,
            "risk_scores": risk_scores,
            "incident_log": incident_log,
            "release_id": release_id,
            "releases_dir": DEFAULT_COMPLIANCE_PATHS["releases_dir"],
            "preprocessed_data": preprocessed_data,
            "warnings": warnings,
        }

    def _calculate_overall_compliance(
        self, article_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate weighted overall compliance score."""
        # Default weights (could also be moved to YAML config)
        weights = {
            "article_9": 0.15,
            "article_10": 0.15,
            "article_11": 0.15,
            "article_12": 0.1,
            "article_13": 0.1,
            "article_14": 0.1,
            "article_15": 0.15,
            "article_16": 0.05,
            "article_17": 0.05,
        }

        # Calculate weighted score
        total_weight = sum(weights.get(a, 0) for a in article_results.keys())
        weighted_sum = sum(
            weights.get(article_id, 0) * result.get("compliance_score", 0)
            for article_id, result in article_results.items()
        )
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Collect findings
        all_findings = []
        for article_id, result in article_results.items():
            for finding in result.get("findings", []):
                if finding.get("type") in ["critical", "warning"]:
                    all_findings.append({**finding, "article": article_id})

        # Sort by severity
        sorted_findings = sorted(
            all_findings, key=lambda f: 0 if f.get("type") == "critical" else 1
        )

        return {
            "overall_compliance_score": overall_score,
            "article_scores": {
                article_id: result.get("compliance_score", 0)
                for article_id, result in article_results.items()
            },
            "critical_findings_count": sum(
                1 for f in all_findings if f.get("type") == "critical"
            ),
            "warning_findings_count": sum(
                1 for f in all_findings if f.get("type") == "warning"
            ),
            "key_findings": sorted_findings[:10],
        }
