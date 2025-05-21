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

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .compliance_constants import DEFAULT_COMPLIANCE_PATHS
from .exceptions import ComplianceDataError
from .schemas import RISK_REGISTER_SCHEMA

# Configure logging
logger = logging.getLogger(__name__)


class ComplianceDataLoader:
    """Handles loading and validation of compliance data."""

    @staticmethod
    def load_risk_register(
        risk_register_path: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Load and validate the risk register from an Excel file.

        Args:
            risk_register_path: Path to the risk register Excel file.
                            If not provided, uses the default path.

        Returns:
            Tuple containing:
            - DataFrame containing the risk register data
            - List of validation warnings (if any)

        Raises:
            ComplianceDataError: If the risk register cannot be loaded or has critical validation issues.
        """
        # Use default path if not provided
        if risk_register_path is None:
            risk_register_path = DEFAULT_COMPLIANCE_PATHS["risk_register"]

        # Check if the file exists
        if not os.path.exists(risk_register_path):
            error_msg = f"Risk register file not found at {risk_register_path}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        try:
            # Load the risk register data
            risk_df = pd.read_excel(risk_register_path)
            logger.info(
                f"Successfully loaded risk register from {risk_register_path}"
            )
        except Exception as e:
            error_msg = f"Failed to load risk register: {str(e)}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        # Validate and standardize the risk register
        warnings = []
        try:
            risk_df, validation_warnings = (
                ComplianceDataLoader._validate_risk_register(risk_df)
            )
            warnings.extend(validation_warnings)
        except Exception as e:
            error_msg = f"Risk register validation failed: {str(e)}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        return risk_df, warnings

    @staticmethod
    def save_risk_register(
        risk_df: pd.DataFrame, risk_register_path: Optional[str] = None
    ) -> None:
        """Save the risk register DataFrame to an Excel file.

        Args:
            risk_df: DataFrame containing the risk register data
            risk_register_path: Path to save the risk register Excel file.
                            If not provided, uses the default path.

        Raises:
            ComplianceDataError: If the risk register cannot be saved.
        """
        # Use default path if not provided
        if risk_register_path is None:
            risk_register_path = DEFAULT_COMPLIANCE_PATHS["risk_register"]

        # Ensure the directory exists
        os.makedirs(os.path.dirname(risk_register_path), exist_ok=True)

        try:
            # Save the risk register data
            risk_df.to_excel(risk_register_path, index=False)
            logger.info(
                f"Successfully saved risk register to {risk_register_path}"
            )
        except Exception as e:
            error_msg = f"Failed to save risk register: {str(e)}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

    @staticmethod
    def load_incident_log(
        incident_log_path: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load and validate the incident log from a JSON file.

        Args:
            incident_log_path: Path to the incident log JSON file.
                            If not provided, uses the default path.

        Returns:
            Tuple containing:
            - List of incident records
            - List of validation warnings (if any)

        Raises:
            ComplianceDataError: If the incident log cannot be loaded or has critical validation issues.
        """
        # Use default path if not provided
        if incident_log_path is None:
            incident_log_path = DEFAULT_COMPLIANCE_PATHS["incident_log"]

        # Check if the file exists
        if not os.path.exists(incident_log_path):
            error_msg = f"Incident log file not found at {incident_log_path}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        try:
            # Load the incident log data
            with open(incident_log_path, "r") as f:
                incidents = json.load(f)
            logger.info(
                f"Successfully loaded incident log from {incident_log_path}"
            )
        except Exception as e:
            error_msg = f"Failed to load incident log: {str(e)}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        # Validate the incident log
        warnings = []
        try:
            incidents, validation_warnings = (
                ComplianceDataLoader._validate_incident_log(incidents)
            )
            warnings.extend(validation_warnings)
        except Exception as e:
            error_msg = f"Incident log validation failed: {str(e)}"
            logger.error(error_msg)
            raise ComplianceDataError(error_msg)

        return incidents, warnings

    @staticmethod
    def load_evaluation_results(
        release_id: str,
        eval_file: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Load evaluation results for a specific release.

        Args:
            release_id: The ID of the release
            eval_file: Optional filename for evaluation results.
                    If not provided, uses the default name.

        Returns:
            Tuple containing:
            - Dictionary with evaluation results
            - List of warnings

        Raises:
            ComplianceDataError: If evaluation results cannot be loaded
        """
        warnings = []

        # Use default filename if not provided
        if eval_file is None:
            eval_file = DEFAULT_COMPLIANCE_PATHS["evaluation_results"]

        # Construct the full path
        releases_dir = DEFAULT_COMPLIANCE_PATHS["releases_dir"]
        file_path = os.path.join(releases_dir, release_id, eval_file)

        # Check if the file exists
        if not os.path.exists(file_path):
            error_msg = f"Evaluation results file not found at {file_path}"
            logger.error(error_msg)
            # Instead of raising an error, return empty results with a warning
            return {"metrics": {}, "fairness": {}}, [error_msg]

        try:
            # Load the evaluation results
            with open(file_path, "r") as f:
                results = yaml.safe_load(f)
            logger.info(
                f"Successfully loaded evaluation results from {file_path}"
            )
        except Exception as e:
            error_msg = f"Failed to load evaluation results: {str(e)}"
            logger.error(error_msg)
            # Instead of raising an error, return empty results with a warning
            return {"metrics": {}, "fairness": {}}, [error_msg]

        return results, warnings

    @staticmethod
    def get_latest_release_id() -> Optional[str]:
        """Get the ID of the latest release by checking the release directories.

        Returns:
            The ID of the latest release, or None if no releases are found.
        """
        releases_dir = DEFAULT_COMPLIANCE_PATHS["releases_dir"]

        # Check if releases directory exists
        if not os.path.exists(releases_dir):
            logger.warning(f"Releases directory not found at {releases_dir}")
            return None

        # Get all subdirectories in the releases directory
        release_dirs = [
            d
            for d in os.listdir(releases_dir)
            if os.path.isdir(os.path.join(releases_dir, d))
        ]

        if not release_dirs:
            logger.warning(f"No release directories found in {releases_dir}")
            return None

        # Get the latest release by modification time
        latest_release = max(
            release_dirs,
            key=lambda d: os.path.getmtime(os.path.join(releases_dir, d)),
        )

        return latest_release

    @staticmethod
    def get_pipeline_log_paths(
        pipeline_name: str, run_id: Optional[str] = None
    ) -> Dict[str, str]:
        """Get the paths to the log files for a specific pipeline run.

        This method retrieves metadata about the logs without saving them again,
        to meet Article 12 (Record Keeping) compliance requirements.

        Args:
            pipeline_name: Name of the pipeline
            run_id: Optional run ID (uses latest if not provided)

        Returns:
            Dictionary with metadata about log files (uri, created, run_id)
        """
        try:
            from zenml.client import Client

            client = Client()
            pipeline = client.get_pipeline(pipeline_name)

            if not pipeline or not pipeline.runs:
                logger.warning(f"No runs found for pipeline {pipeline_name}")
                return {}

            # Find the specific run or use the latest
            if run_id:
                run = next(
                    (r for r in pipeline.runs if str(r.id) == run_id), None
                )
                if not run:
                    logger.warning(
                        f"Run ID {run_id} not found for pipeline {pipeline_name}"
                    )
                    return {}
            else:
                # Use the latest run
                run = pipeline.runs[-1]
                run_id = str(run.id)

            # Get log metadata without saving the content
            log_info = run.logs  # Access the LogsResponseBody object

            if hasattr(log_info, "uri") and log_info.uri:
                result = {
                    "log_uri": log_info.uri,
                    "pipeline_name": pipeline_name,
                    "run_id": run_id,
                }

                if hasattr(log_info, "created"):
                    result["created"] = str(log_info.created)

                logger.info(
                    f"Found log file for pipeline {pipeline_name} at {log_info.uri}"
                )
                return result
            else:
                logger.warning(f"No log URI found for run {run_id}")
                return {}

        except Exception as e:
            logger.error(
                f"Failed to get log paths for pipeline {pipeline_name}: {e}"
            )
            return {}

    @staticmethod
    def preprocess_compliance_data(risk_df: pd.DataFrame) -> Dict[str, Any]:
        """Preprocess risk register data for compliance calculations.

        Args:
            risk_df: DataFrame containing the risk register data

        Returns:
            Dictionary with preprocessed compliance data
        """
        compliance_data = {}

        # Calculate risk register metrics
        compliance_data["risks_count"] = len(risk_df)

        # Calculate percentage of risks with defined mitigation measures
        has_mitigation = risk_df["Mitigation"].apply(
            lambda x: isinstance(x, str) and len(x.strip()) > 0
        )
        compliance_data["mitigated_risks_percentage"] = has_mitigation.mean()

        # Calculate percentage of completed mitigations
        if "Mitigation_status" in risk_df.columns:
            completed_mitigations = (
                risk_df["Mitigation_status"] == "COMPLETED"
            ).sum()
            compliance_data["completed_mitigations_percentage"] = (
                completed_mitigations / len(risk_df) if len(risk_df) > 0 else 0
            )
        else:
            compliance_data["completed_mitigations_percentage"] = 0

        # Group risks by article
        if "Article" in risk_df.columns:
            article_counts = risk_df["Article"].value_counts().to_dict()
            compliance_data["risks_by_article"] = article_counts

            # Calculate completion percentage by article
            article_completion = {}
            for article in article_counts.keys():
                article_risks = risk_df[risk_df["Article"] == article]
                completed = (
                    article_risks["Mitigation_status"] == "COMPLETED"
                ).sum()
                article_completion[article] = (
                    completed / len(article_risks)
                    if len(article_risks) > 0
                    else 0
                )

            compliance_data["completion_by_article"] = article_completion

        # Calculate average risk score
        compliance_data["average_risk_overall"] = risk_df[
            "Risk_overall"
        ].mean()

        # Calculate distribution by risk category
        if "Risk_category" in risk_df.columns:
            category_counts = risk_df["Risk_category"].value_counts().to_dict()
            compliance_data["risks_by_category"] = category_counts

        return compliance_data

    @staticmethod
    def _validate_risk_register(
        risk_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate the risk register DataFrame against the schema and standardize values.

        Args:
            risk_df: DataFrame containing the risk register data

        Returns:
            Tuple containing:
            - Validated and standardized DataFrame
            - List of validation warnings

        Raises:
            ComplianceDataError: If critical validation issues are found
        """
        warnings = []

        # Check required columns
        missing_columns = [
            col
            for col in RISK_REGISTER_SCHEMA["required_columns"]
            if col not in risk_df.columns
        ]
        if missing_columns:
            # Check if these are the new columns we're adding
            new_columns = ["Article", "Mitigation_status", "Review_date"]
            if all(col in new_columns for col in missing_columns):
                # Add missing columns with default values
                for col in missing_columns:
                    if col == "Article":
                        risk_df[col] = (
                            "article_9"  # Default to article 9 (risk management)
                        )
                    elif col == "Mitigation_status":
                        risk_df[col] = risk_df[
                            "Status"
                        ]  # Copy from Status initially
                    elif col == "Review_date":
                        risk_df[col] = datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )

                warnings.append(
                    f"Added new columns to risk register: {', '.join(missing_columns)}"
                )
            else:
                critical_missing = [
                    col for col in missing_columns if col not in new_columns
                ]
                if critical_missing:
                    error_msg = f"Risk register is missing required columns: {', '.join(critical_missing)}"
                    logger.error(error_msg)
                    raise ComplianceDataError(error_msg)

        # Standardize column values
        # Convert Status and Risk_category to uppercase
        if "Status" in risk_df.columns:
            risk_df["Status"] = risk_df["Status"].str.upper()

            # Validate Status values
            invalid_statuses = [
                status
                for status in risk_df["Status"].unique()
                if status not in RISK_REGISTER_SCHEMA["valid_values"]["Status"]
            ]
            if invalid_statuses:
                warnings.append(
                    f"Found invalid Status values: {invalid_statuses}. "
                    f"Valid values are: {RISK_REGISTER_SCHEMA['valid_values']['Status']}"
                )
                # Standardize invalid values
                risk_df["Status"] = risk_df["Status"].apply(
                    lambda x: "PENDING"
                    if x not in RISK_REGISTER_SCHEMA["valid_values"]["Status"]
                    else x
                )

        if "Mitigation_status" in risk_df.columns:
            risk_df["Mitigation_status"] = risk_df[
                "Mitigation_status"
            ].str.upper()

            # Validate Mitigation_status values
            invalid_mitigation_statuses = [
                status
                for status in risk_df["Mitigation_status"].unique()
                if status
                not in RISK_REGISTER_SCHEMA["valid_values"][
                    "Mitigation_status"
                ]
            ]
            if invalid_mitigation_statuses:
                warnings.append(
                    f"Found invalid Mitigation_status values: {invalid_mitigation_statuses}. "
                    f"Valid values are: {RISK_REGISTER_SCHEMA['valid_values']['Mitigation_status']}"
                )
                # Standardize invalid values
                risk_df["Mitigation_status"] = risk_df[
                    "Mitigation_status"
                ].apply(
                    lambda x: "PENDING"
                    if x
                    not in RISK_REGISTER_SCHEMA["valid_values"][
                        "Mitigation_status"
                    ]
                    else x
                )

        if "Risk_category" in risk_df.columns:
            risk_df["Risk_category"] = risk_df["Risk_category"].str.upper()

            # Validate Risk_category values
            invalid_categories = [
                category
                for category in risk_df["Risk_category"].unique()
                if category
                not in RISK_REGISTER_SCHEMA["valid_values"]["Risk_category"]
            ]
            if invalid_categories:
                warnings.append(
                    f"Found invalid Risk_category values: {invalid_categories}. "
                    f"Valid values are: {RISK_REGISTER_SCHEMA['valid_values']['Risk_category']}"
                )
                # Standardize invalid values (map to closest valid value)
                category_mapping = {
                    "LOW RISK": "LOW",
                    "MEDIUM RISK": "MEDIUM",
                    "HIGH RISK": "HIGH",
                    "CRITICAL RISK": "CRITICAL",
                    "L": "LOW",
                    "M": "MEDIUM",
                    "H": "HIGH",
                    "C": "CRITICAL",
                }
                risk_df["Risk_category"] = risk_df["Risk_category"].apply(
                    lambda x: category_mapping.get(x, "MEDIUM")
                    if x
                    not in RISK_REGISTER_SCHEMA["valid_values"][
                        "Risk_category"
                    ]
                    else x
                )

        # Validate Article values if present
        if "Article" in risk_df.columns:
            # Map any invalid article values to a valid article based on risk description
            invalid_articles = [
                article
                for article in risk_df["Article"].unique()
                if article
                not in RISK_REGISTER_SCHEMA["valid_values"]["Article"]
            ]
            if invalid_articles:
                warnings.append(
                    f"Found invalid Article values: {invalid_articles}. "
                    f"Valid values are: {RISK_REGISTER_SCHEMA['valid_values']['Article']}"
                )

                # Map invalid articles based on risk description
                def map_article(row):
                    if (
                        row["Article"]
                        in RISK_REGISTER_SCHEMA["valid_values"]["Article"]
                    ):
                        return row["Article"]

                    risk_desc = (
                        row["Risk_description"].lower()
                        if isinstance(row.get("Risk_description"), str)
                        else ""
                    )

                    # Simple keyword mapping
                    if any(
                        kw in risk_desc
                        for kw in [
                            "bias",
                            "fair",
                            "protected",
                            "discrimination",
                        ]
                    ):
                        return "article_10"  # Data governance
                    elif any(
                        kw in risk_desc
                        for kw in ["accuracy", "robustness", "performance"]
                    ):
                        return "article_15"  # Accuracy and robustness
                    elif any(
                        kw in risk_desc for kw in ["document", "documentation"]
                    ):
                        return "article_11"  # Technical documentation
                    elif any(kw in risk_desc for kw in ["monitor", "drift"]):
                        return "article_17"  # Post-market monitoring
                    elif any(
                        kw in risk_desc
                        for kw in ["human", "oversight", "review"]
                    ):
                        return "article_14"  # Human oversight
                    elif any(
                        kw in risk_desc
                        for kw in ["transparent", "transparency", "explain"]
                    ):
                        return "article_13"  # Transparency
                    elif any(
                        kw in risk_desc for kw in ["quality", "management"]
                    ):
                        return "article_16"  # Quality management
                    elif any(
                        kw in risk_desc for kw in ["record", "log", "audit"]
                    ):
                        return "article_12"  # Record keeping
                    else:
                        return "article_9"  # Default to risk management

                risk_df["Article"] = risk_df.apply(map_article, axis=1)

        # Ensure datetime format for Timestamp and Review_date
        for date_col in ["Timestamp", "Review_date"]:
            if date_col in risk_df.columns:
                # Check if dates are in ISO format
                invalid_dates = []
                for i, date_str in enumerate(risk_df[date_col]):
                    if not isinstance(date_str, str):
                        risk_df.at[i, date_col] = datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )
                        invalid_dates.append(f"Row {i + 1}")
                    elif not date_str.startswith(
                        "20"
                    ):  # Simple check for year format
                        try:
                            # Try to parse and reformat
                            parsed_date = pd.to_datetime(date_str)
                            risk_df.at[i, date_col] = parsed_date.strftime(
                                "%Y-%m-%dT%H:%M:%S.%f"
                            )
                        except Exception:
                            risk_df.at[i, date_col] = datetime.now().strftime(
                                "%Y-%m-%dT%H:%M:%S.%f"
                            )
                            invalid_dates.append(f"Row {i + 1}")

                if invalid_dates:
                    warnings.append(
                        f"Fixed invalid {date_col} formats in rows: {', '.join(invalid_dates[:5])}"
                        f"{' and more...' if len(invalid_dates) > 5 else ''}"
                    )

        return risk_df, warnings

    @staticmethod
    def _validate_incident_log(
        incidents: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate the incident log records and standardize values.

        Args:
            incidents: List of incident records

        Returns:
            Tuple containing:
            - Validated and standardized incident records
            - List of validation warnings

        Raises:
            ComplianceDataError: If critical validation issues are found
        """
        warnings = []

        # Define required fields for incident records
        required_fields = [
            "incident_id",
            "timestamp",
            "severity",
            "description",
            "source",
        ]

        valid_severity_values = ["low", "medium", "high", "critical"]

        for i, incident in enumerate(incidents):
            # Check for missing required fields
            missing_fields = [
                field for field in required_fields if field not in incident
            ]
            if missing_fields:
                warnings.append(
                    f"Incident {i + 1} is missing required fields: {', '.join(missing_fields)}"
                )
                # Add missing fields with default values
                for field in missing_fields:
                    if field == "incident_id":
                        incident[field] = (
                            f"incident_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')}"
                        )
                    elif field == "timestamp":
                        incident[field] = datetime.now().strftime(
                            "%Y-%m-%dT%H:%M:%S.%f"
                        )
                    elif field == "severity":
                        incident[field] = "medium"
                    elif field == "description":
                        incident[field] = "Unknown incident"
                    elif field == "source":
                        incident[field] = "unknown"

            # Standardize severity values
            if "severity" in incident:
                orig_severity = incident["severity"]
                # Convert to lowercase
                incident["severity"] = (
                    incident["severity"].lower()
                    if isinstance(incident["severity"], str)
                    else "medium"
                )

                # Check if severity is valid
                if incident["severity"] not in valid_severity_values:
                    warnings.append(
                        f"Incident {i + 1} has invalid severity value: {orig_severity}. "
                        f"Valid values are: {valid_severity_values}"
                    )
                    # Map invalid severity values
                    severity_mapping = {
                        "l": "low",
                        "m": "medium",
                        "h": "high",
                        "c": "critical",
                        "severe": "critical",
                        "important": "high",
                        "warning": "medium",
                        "info": "low",
                    }
                    incident["severity"] = severity_mapping.get(
                        incident["severity"].lower(), "medium"
                    )

            # Add resolution_status field if not present
            if "resolution_status" not in incident:
                incident["resolution_status"] = "open"
                warnings.append(
                    f"Added missing resolution_status field to incident {i + 1}"
                )

            # Add article field if not present
            if "article" not in incident:
                # Determine article based on description or source
                description = incident.get("description", "").lower()
                source = incident.get("source", "").lower()

                # Simple keyword mapping
                if any(
                    kw in description or kw in source
                    for kw in ["bias", "fair", "protected"]
                ):
                    incident["article"] = "article_10"  # Data governance
                elif any(
                    kw in description or kw in source
                    for kw in ["accuracy", "performance"]
                ):
                    incident["article"] = (
                        "article_15"  # Accuracy and robustness
                    )
                elif any(
                    kw in description or kw in source
                    for kw in ["monitor", "drift"]
                ):
                    incident["article"] = (
                        "article_17"  # Post-market monitoring
                    )
                else:
                    incident["article"] = (
                        "article_9"  # Default to risk management
                    )

                warnings.append(
                    f"Added missing article field to incident {i + 1}"
                )

        return incidents, warnings
