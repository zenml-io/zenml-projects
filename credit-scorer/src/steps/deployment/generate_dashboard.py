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

from pathlib import Path
from typing import Annotated

import pandas as pd
from streamlit_app.data.compliance_utils import get_compliance_results
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from src.constants import (
    COMPLIANCE_DASHBOARD_HTML_NAME,
    RISK_REGISTER_PATH,
)
from src.utils.visualizations.compliance_dashboard import (
    create_compliance_dashboard_artifact,
)

logger = get_logger(__name__)


@step
def generate_compliance_dashboard(
    run_release_dir: str,
) -> Annotated[HTMLString, COMPLIANCE_DASHBOARD_HTML_NAME]:
    """Generate a compliance dashboard HTML artifact.

    This step creates an HTML dashboard visualization of the compliance status
    for the EU AI Act requirements. The dashboard can be saved as a ZenML artifact
    and stored as part of the model release.

    Returns:
        HTML dashboard as a ZenML HTMLString artifact
    """
    logger.info("Generating compliance dashboard HTML artifact")

    # Get compliance results using current run ID
    compliance_results = get_compliance_results(
        run_release_dir=run_release_dir
    )

    # Load risk data if available
    risk_df = None
    try:
        risk_register_path = Path(RISK_REGISTER_PATH)
        if risk_register_path.exists():
            risk_df = pd.read_excel(risk_register_path)
            logger.info(f"Loaded risk data with {len(risk_df)} entries")
    except Exception as e:
        logger.warning(f"Failed to load risk data: {e}")

    # Create incident data (in a real scenario, this would be loaded from a database)
    # For simplicity, we'll create a sample DataFrame
    incident_df = pd.DataFrame(
        [
            {
                "timestamp": "2025-05-15 14:30",
                "description": "Data drift detected in age distribution",
                "severity": "MEDIUM",
            },
            {
                "timestamp": "2025-05-10 09:15",
                "description": "Potential age bias in model predictions",
                "severity": "HIGH",
            },
        ]
    )

    # Create the HTML dashboard artifact (returns HTMLString)
    dashboard_html = create_compliance_dashboard_artifact(
        compliance_results=compliance_results,
        risk_df=risk_df,
        incident_df=incident_df,
    )

    # Save the HTML to the release directory as well
    # Make sure the release directory exists
    release_dir = Path(run_release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    # Save the HTML file
    html_path = release_dir / "compliance_dashboard.html"
    with open(html_path, "w") as f:
        f.write(dashboard_html)

    logger.info(f"Saved compliance dashboard to {html_path}")

    return dashboard_html
