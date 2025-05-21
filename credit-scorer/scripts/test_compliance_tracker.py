#!/usr/bin/env python3
# Apache Software License 2.0

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the compliance module
from src.utils.compliance import DEFAULT_COMPLIANCE_PATHS, ComplianceDataError
from src.utils.compliance.compliance_calculator import ComplianceCalculator, calculate_compliance
from src.utils.compliance.data_loader import ComplianceDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_load_risk_register():
    """Test loading and validating the risk register."""
    data_loader = ComplianceDataLoader()
    try:
        logger.info("Testing ComplianceDataLoader.load_risk_register()...")
        risk_df, warnings = data_loader.load_risk_register()

        logger.info(f"Successfully loaded risk register with {len(risk_df)} risks")
        if warnings:
            logger.warning(f"Warnings: {len(warnings)}")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        return risk_df
    except ComplianceDataError as e:
        logger.error(f"Failed to load risk register: {str(e)}")
        return None


def test_load_incident_log():
    """Test loading and validating the incident log."""
    data_loader = ComplianceDataLoader()
    try:
        logger.info("\nTesting ComplianceDataLoader.load_incident_log()...")
        incidents, warnings = data_loader.load_incident_log()

        logger.info(f"Successfully loaded incident log with {len(incidents)} incidents")
        if warnings:
            logger.warning(f"Warnings: {len(warnings)}")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        # Print preview
        # logger.info("\nIncident log preview:")
        # for incident in incidents[:3]:
        #     print(json.dumps(incident, indent=2))
        #     print("---")

        return incidents
    except ComplianceDataError as e:
        logger.error(f"Failed to load incident log: {str(e)}")
        return None


def debug_article_9_config():
    """Debug Article 9 configuration and data."""
    try:
        logger.info("\n=== DEBUGGING ARTICLE 9 ===")

        # 1. Load config and check Article 9 definition
        config_path = Path(__file__).parent.parent / "src/utils/compliance/compliance_articles.yaml"

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return

        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Find Article 9 config
        article_9_config = None
        for article in config.get("articles", []):
            if article.get("id") == "article_9":
                article_9_config = article
                break

        if not article_9_config:
            logger.error("Article 9 not found in config")
            return

        # logger.info("Article 9 config:")
        # import json

        # print(json.dumps(article_9_config, indent=2))

        # 2. Load and inspect risk register
        data_loader = ComplianceDataLoader()
        risk_df, _ = data_loader.load_risk_register()

        logger.info(f"\nRisk register shape: {risk_df.shape}")
        logger.info(f"Risk register columns: {list(risk_df.columns)}")

        # Check Article column
        if "Article" in risk_df.columns:
            article_counts = risk_df["Article"].value_counts()
            logger.info(f"Article distribution: {dict(article_counts)}")
            article_9_risks = risk_df[risk_df["Article"] == "article_9"]
            logger.info(f"Article 9 risks: {len(article_9_risks)}")
        else:
            logger.warning("No 'Article' column found in risk register")

        # Check Mitigation column
        if "Mitigation" in risk_df.columns:
            has_mitigation = risk_df["Mitigation"].apply(
                lambda x: bool(str(x).strip()) if pd.notna(x) else False
            )
            logger.info(f"Risks with mitigation: {has_mitigation.sum()}/{len(risk_df)}")
        else:
            logger.warning("No 'Mitigation' column found")

        # Check other relevant columns
        for col in ["Mitigation_status", "Review_date"]:
            if col in risk_df.columns:
                logger.info(f"{col} present: {risk_df[col].notna().sum()}/{len(risk_df)} non-null")
            else:
                logger.warning(f"No '{col}' column found")

    except Exception as e:
        logger.error(f"Debug failed: {e}")
        import traceback

        traceback.print_exc()


def test_config_driven_calculator():
    """Test the ComplianceCalculator directly."""
    try:
        logger.info("\nTesting ComplianceCalculator...")

        # Load data
        data_loader = ComplianceDataLoader()
        risk_df, _ = data_loader.load_risk_register()
        release_id = data_loader.get_latest_release_id()

        # Debug data being passed
        logger.info(f"Risk register shape: {risk_df.shape}")
        logger.info(f"Risk register columns: {list(risk_df.columns)}")
        if "Article" in risk_df.columns:
            logger.info(f"Article 9 risks: {len(risk_df[risk_df['Article'] == 'article_9'])}")

        # Prepare data
        data = {
            "risk_register": risk_df,
            "release_id": release_id,
            "releases_dir": DEFAULT_COMPLIANCE_PATHS["releases_dir"],
        }

        # Add risk scores for more complete testing
        risk_scores_path = os.path.join(
            DEFAULT_COMPLIANCE_PATHS["releases_dir"],
            release_id,
            "risk_scores.yaml",
        )
        if os.path.exists(risk_scores_path):
            try:
                with open(risk_scores_path, "r") as f:
                    risk_scores = yaml.safe_load(f)
                data["risk_scores"] = risk_scores
                logger.info(f"Added risk scores from {risk_scores_path}")
            except Exception as e:
                logger.warning(f"Failed to load risk scores: {e}")

        # Test Article 9 calculator
        config_path = Path(__file__).parent.parent / "src/utils/compliance/compliance_articles.yaml"

        # Verify config exists
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return None

        calculator = ComplianceCalculator(str(config_path), "article_9")
        result = calculator.calculate(**data)

        # Print Article 9 specific results
        logger.info("Article 9 Compliance Results:")
        logger.info(f"Compliance Score: {result['compliance_score']:.2f}%")

        # Log detailed metrics for debugging
        logger.info("Detailed Metrics:")
        score_keys = [k for k in result["metrics"].keys() if k.endswith("_score")]
        for key in score_keys:
            name = key.replace("_score", "")
            logger.info(f"  - {name}: {result['metrics'][key]:.2f}%")

        logger.info("Findings:")
        for i, finding in enumerate(result["findings"], 1):
            logger.info(f"  {i}. {finding['type'].upper()}: {finding['message']}")
            if "recommendation" in finding:
                logger.info(f"     Recommendation: {finding['recommendation']}")

        return result
    except Exception as e:
        logger.error(f"Failed to test ComplianceCalculator: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def test_calculate_compliance():
    """Test calculating compliance for all articles."""
    try:
        logger.info("\nTesting calculate_compliance()...")

        # Calculate compliance for all articles
        results = calculate_compliance()

        logger.info("Successfully calculated compliance")
        logger.info(
            f"Overall compliance score: {results['overall']['overall_compliance_score']:.1f}%"
        )

        # Print per-article scores
        logger.info("\nCompliance scores by article:")
        for article_id, score in results["overall"]["article_scores"].items():
            logger.info(f"  - {article_id}: {score:.1f}%")

        # Print key findings
        logger.info("\nKey findings:")
        for i, finding in enumerate(results["overall"]["key_findings"][:5]):
            logger.info(
                f"  {i + 1}. [{finding['type'].upper()}] {finding['article']}: {finding['message']}"
            )

        return results
    except Exception as e:
        logger.error(f"Failed to calculate compliance: {str(e)}")


def main():
    """Run all tests for the compliance tracker."""
    logger.info("=== Running Compliance Tracker Tests ===")

    # Debug to understand data structures
    debug_article_9_config()

    # Test loading risk register
    test_load_risk_register()

    # Test loading incident log
    test_load_incident_log()

    # Test ComplianceCalculator directly
    test_config_driven_calculator()

    # Test calculating compliance
    test_calculate_compliance()

    logger.info("=== All Tests Complete ===")


if __name__ == "__main__":
    main()
