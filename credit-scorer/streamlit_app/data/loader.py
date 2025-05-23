"""Data loading functions for the dashboard."""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit_app.config import (
    INCIDENT_LOG_PATH,
    RELEASES_DIR,
    RISK_REGISTER_PATH,
    SAMPLE_INPUTS_PATH,
)


def load_risk_register():
    """Load the risk register from the default location or most recent file."""
    risk_register_path = Path(RISK_REGISTER_PATH)

    try:
        excel_data = pd.read_excel(risk_register_path, sheet_name=None)

        # Process each sheet - normalize column names to lowercase
        for sheet_name, df in excel_data.items():
            excel_data[sheet_name].columns = [
                col.lower() for col in df.columns
            ]

            # Add article column if needed for compliance tracking
            if (
                sheet_name == "Risks"
                and "article" not in excel_data[sheet_name].columns
            ):
                # Try to derive article from category if available
                if "category" in excel_data[sheet_name].columns:
                    # Extract article numbers (e.g., "Art 9 requirement" -> "9")
                    excel_data[sheet_name]["article"] = (
                        excel_data[sheet_name]["category"]
                        .astype(str)
                        .str.extract(
                            r"(?:art(?:icle)?\s*)?(\d+)", flags=re.IGNORECASE
                        )
                        .fillna("")
                    )
                # If no category column, try description
                elif "risk_description" in excel_data[sheet_name].columns:
                    # Extract article numbers from description
                    excel_data[sheet_name]["article"] = (
                        excel_data[sheet_name]["risk_description"]
                        .astype(str)
                        .str.extract(
                            r"(?:art(?:icle)?\s*)?(\d+)", flags=re.IGNORECASE
                        )
                        .fillna("")
                    )

        return excel_data
    except Exception as e:
        st.error(f"Error loading risk register: {e}")
        return None


def load_incident_log():
    """Load the incident log from the default location."""
    try:
        with open(INCIDENT_LOG_PATH, "r") as f:
            incidents = json.load(f)

        # Convert to DataFrame for easier manipulation
        if isinstance(incidents, list):
            return pd.DataFrame(incidents)
        else:
            return pd.DataFrame(incidents.get("incidents", []))
    except Exception as e:
        st.warning(f"Error loading incident log: {e}")
        return pd.DataFrame()


def load_latest_release_info():
    """Load information about the latest release directory."""
    try:
        # Find the most recent release directory
        release_dirs = sorted(
            [d for d in RELEASES_DIR.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not release_dirs:
            return None

        latest_release = release_dirs[0]

        # Get all files in the release directory
        files = list(latest_release.glob("*"))

        # Check for required compliance files
        required_files = {
            "evaluation_results.yaml": False,
            "risk_scores.yaml": False,
            "monitoring_plan.json": False,
            "sbom.json": False,
            "annex_iv.md": False,
        }

        for file_path in files:
            if file_path.name in required_files:
                required_files[file_path.name] = True

        missing_files = [
            name for name, found in required_files.items() if not found
        ]

        # Return relevant information
        return {
            "path": latest_release,
            "id": latest_release.name,
            "date": datetime.fromtimestamp(
                latest_release.stat().st_mtime
            ).strftime("%Y-%m-%d %H:%M"),
            "files": files,
            "missing_files": missing_files,
            "is_complete": len(missing_files) == 0,
        }
    except Exception as e:
        st.error(f"Error loading release info: {e}")
        return None


def load_latest_annex_iv():
    """Load the most recent Annex IV document."""
    release_info = load_latest_release_info()

    if not release_info:
        return None, None

    try:
        # Find the Annex IV document
        annex_files = list(release_info["path"].glob("annex_iv.md"))
        if not annex_files:
            st.warning(
                f"No Annex IV document found in {release_info['path']}."
            )
            return None, None

        # Load the first Annex IV document found
        annex_path = annex_files[0]
        with open(annex_path, "r") as f:
            content = f.read()

        return content, annex_path

    except Exception as e:
        st.error(f"Error loading Annex IV document: {e}")
        return None, None


def load_latest_whylogs_profile():
    """Load the path to the most recent whylogs profile HTML."""
    release_info = load_latest_release_info()

    if not release_info:
        return None

    try:
        # Find the whylogs profile document
        profile_files = list(release_info["path"].glob("whylogs_profile.html"))
        if not profile_files:
            return None

        # Return the path to the first profile found
        return profile_files[0]

    except Exception as e:
        st.error(f"Error loading whylogs profile: {e}")
        return None


def load_manual_inputs():
    """Load the manual inputs for the Annex IV document."""
    try:
        with open(SAMPLE_INPUTS_PATH, "r") as f:
            manual_inputs = json.load(f)

        # Auto-populate frameworks from requirements.txt if not already set
        if (
            "frameworks" not in manual_inputs
            or not manual_inputs["frameworks"]
        ):
            manual_inputs["frameworks"] = parse_requirements_txt()

        return manual_inputs
    except Exception as e:
        st.error(f"Error loading manual inputs: {e}")
        return {}


def save_manual_inputs(manual_inputs):
    """Save the manual inputs to the sample inputs file."""
    try:
        with open(SAMPLE_INPUTS_PATH, "w") as f:
            json.dump(manual_inputs, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving manual inputs: {e}")
        return False


def parse_requirements_txt(requirements_path="requirements.txt"):
    """Parse requirements.txt and extract package versions."""
    frameworks = {}

    try:
        with open(requirements_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse package==version or package>=version patterns
            if "==" in line:
                package, version = line.split("==", 1)
                frameworks[package.strip()] = version.strip()
            elif ">=" in line:
                package, version = line.split(">=", 1)
                frameworks[package.strip()] = f">={version.strip()}"
            elif "~=" in line:
                package, version = line.split("~=", 1)
                frameworks[package.strip()] = f"~={version.strip()}"
            else:
                # Handle cases like "package" without version
                package = line
                frameworks[package.strip()] = "latest"

    except FileNotFoundError:
        st.warning(f"requirements.txt not found at {requirements_path}")
        return {}
    except Exception as e:
        st.error(f"Error parsing requirements.txt: {e}")
        return {}

    return frameworks
