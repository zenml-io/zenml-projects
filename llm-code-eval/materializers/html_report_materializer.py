"""Custom materializer for CodeEvalReport.

Extends PydanticMaterializer (free save/load for Pydantic models)
and adds HTML visualization for the ZenML dashboard.
"""

from __future__ import annotations

import os
from typing import Dict

from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer

from utils.scoring import CodeEvalReport


class HTMLReportMaterializer(PydanticMaterializer):
    """Materializer that persists CodeEvalReport + renders HTML in dashboard."""

    ASSOCIATED_TYPES = (CodeEvalReport,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: CodeEvalReport
    ) -> Dict[str, VisualizationType]:
        """Write HTML report for ZenML dashboard rendering."""
        html_path = os.path.join(self.uri, "report.html")
        with fileio.open(html_path, "w") as f:
            f.write(data.report_html)

        return {html_path: VisualizationType.HTML}
