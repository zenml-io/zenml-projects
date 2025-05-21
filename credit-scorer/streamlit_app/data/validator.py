"""Validation utilities for compliance files and data."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st
import yaml

from streamlit_app.config import (
    RELEASES_DIR,
    RISK_REGISTER_PATH,
)

# Set up logging
logger = logging.getLogger(__name__)


def validate_yaml_file(
    filepath: Union[str, Path],
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Validate and load a YAML file.

    Args:
        filepath: Path to the YAML file

    Returns:
        Tuple of (success status, loaded data, list of validation errors)
    """
    errors = []
    data = None

    # Check file exists
    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return False, None, errors

    # Check file extension
    if not str(filepath).lower().endswith((".yaml", ".yml")):
        errors.append(f"File is not a YAML file: {filepath}")

    # Try to load and parse the file
    try:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return True, data, errors
    except yaml.YAMLError as e:
        errors.append(f"Error parsing YAML: {e}")
        return False, None, errors
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, None, errors


def validate_json_file(
    filepath: Union[str, Path],
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """Validate and load a JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        Tuple of (success status, loaded data, list of validation errors)
    """
    errors = []
    data = None

    # Check file exists
    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return False, None, errors

    # Check file extension
    if not str(filepath).lower().endswith(".json"):
        errors.append(f"File is not a JSON file: {filepath}")

    # Try to load and parse the file
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return True, data, errors
    except json.JSONDecodeError as e:
        errors.append(f"Error parsing JSON: {e}")
        return False, None, errors
    except Exception as e:
        errors.append(f"Error reading file: {e}")
        return False, None, errors


def validate_risk_register(
    filepath: Union[str, Path] = RISK_REGISTER_PATH,
) -> Tuple[bool, List[str]]:
    """Validate the risk register file.

    Args:
        filepath: Path to the risk register Excel file

    Returns:
        Tuple of (success status, list of validation errors)
    """
    errors = []

    # Check file exists
    if not os.path.exists(filepath):
        errors.append(f"Risk register file not found: {filepath}")
        return False, errors

    # Check file extension
    if not str(filepath).lower().endswith((".xlsx", ".xls")):
        errors.append(f"File is not an Excel file: {filepath}")

    # Try to load and validate the file
    try:
        excel_data = pd.read_excel(filepath, sheet_name=None)

        # Check required sheets
        if "Risks" not in excel_data:
            errors.append("Required sheet 'Risks' not found in risk register")
            return False, errors

        # Check required columns in Risks sheet
        risk_df = excel_data["Risks"]
        required_columns = ["Risk_description", "Mitigation", "Status"]

        # Normalize column names to lowercase for case-insensitive comparison
        risk_columns = [col.lower() for col in risk_df.columns]
        missing_columns = [col for col in required_columns if col.lower() not in risk_columns]

        if missing_columns:
            errors.append(
                f"Required columns missing from risk register: {', '.join(missing_columns)}"
            )

        return len(errors) == 0, errors

    except Exception as e:
        errors.append(f"Error validating risk register: {e}")
        return False, errors


def validate_release_directory(
    release_id: str,
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """Validate a release directory and its compliance files.

    Args:
        release_id: ID of the release to validate

    Returns:
        Tuple of (success status, dictionary of validation results by file)
    """
    release_dir = Path(RELEASES_DIR) / release_id
    validation_results = {}

    # Check directory exists
    if not release_dir.exists() or not release_dir.is_dir():
        return False, {
            "directory": {
                "exists": False,
                "errors": ["Release directory not found"],
            }
        }

    validation_results["directory"] = {"exists": True, "errors": []}

    # Required files and their validation functions
    required_files = {
        "evaluation_results.yaml": validate_yaml_file,
        "risk_scores.yaml": validate_yaml_file,
        "monitoring_plan.json": validate_json_file,
        "sbom.json": validate_json_file,
        "annex_iv_cs_deployment.md": lambda f: (
            os.path.exists(f),
            None,
            [] if os.path.exists(f) else ["File not found"],
        ),
    }

    # Validate each required file
    all_valid = True
    for filename, validator_func in required_files.items():
        file_path = release_dir / filename

        # Run validation
        is_valid, data, errors = validator_func(file_path)
        all_valid = all_valid and is_valid

        # Store validation results
        validation_results[filename] = {
            "exists": os.path.exists(file_path),
            "valid": is_valid,
            "errors": errors,
            "data": data,
        }

    return all_valid, validation_results


def show_validation_summary(
    validation_results: Dict[str, Dict[str, Any]],
) -> None:
    """Display a summary of validation results in the Streamlit UI.

    Args:
        validation_results: Validation results from validate_release_directory
    """
    # Count issues
    missing_files = [
        name
        for name, result in validation_results.items()
        if name != "directory" and not result.get("exists", False)
    ]

    invalid_files = [
        name
        for name, result in validation_results.items()
        if name != "directory" and result.get("exists", False) and not result.get("valid", False)
    ]

    # Show summary
    if not missing_files and not invalid_files:
        st.success("✅ All compliance files are present and valid")
    else:
        if missing_files:
            st.warning(f"⚠️ Missing files: {', '.join(missing_files)}")

        if invalid_files:
            st.error(f"❌ Invalid files: {', '.join(invalid_files)}")

            # Show detailed errors for invalid files
            with st.expander("Show validation errors"):
                for name in invalid_files:
                    if name in validation_results:
                        errors = validation_results[name].get("errors", [])
                        if errors:
                            st.write(f"**{name}**:")
                            for error in errors:
                                st.write(f"- {error}")
