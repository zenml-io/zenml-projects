"""Configuration settings for the Streamlit dashboard application."""

from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "src" / "data"

# Constants for file paths
RISK_REGISTER_PATH = DOCS_DIR / "risk" / "risk_register.xlsx"
RELEASES_DIR = DOCS_DIR / "releases"
TEMPLATE_DIR = DOCS_DIR / "templates"
SAMPLE_INPUTS_PATH = TEMPLATE_DIR / "sample_inputs.json"
INCIDENT_LOG_PATH = DOCS_DIR / "risk" / "incident_log.json"

# EU AI Act article descriptions
EXPECTED_ARTICLES = {
    "9": "Risk Management",
    "10": "Data Governance",
    "11": "Technical Documentation",
    "13": "Transparency",
    "14": "Human Oversight",
    "15": "Accuracy & Robustness",
    "16": "Post-market Monitoring",
    "17": "Incident Reporting",
}

# Theme settings and styling
PRIMARY_COLOR = (
    "#1F4E79"  # Dark blue - professional look for financial industry
)
SECONDARY_COLOR = "#4F81BD"  # Medium blue
ACCENT_COLOR = "#D9E1F2"  # Light blue background
TEXT_COLOR = "#333333"  # Dark gray for text
RISK_COLORS = {"HIGH": "#D64045", "MEDIUM": "#FFB30F", "LOW": "#478C5C"}


# Runtime storage for asset rendering
class AssetRegistry:
    def __init__(self):
        self.images = []  # List to store image info

    def reset(self):
        self.images = []


# Global registry for assets to render
ASSETS_TO_RENDER = AssetRegistry()
