import os
import io
from typing import Dict

import shap
import matplotlib.pyplot as plt

from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

# Custom class to hold SHAP visualization data
class SHAPVisualization:
    def __init__(self, shap_values, feature_names):
        self.shap_values = shap_values
        self.feature_names = feature_names


# Custom materializer for SHAPVisualization
class SHAPVisualizationMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (SHAPVisualization,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA_ANALYSIS

    def save_visualizations(
            self, data: SHAPVisualization
    ) -> Dict[str, VisualizationType]:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(data.shap_values, feature_names=data.feature_names, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        visualization_path = os.path.join(self.uri, "shap_summary_plot.png")
        with fileio.open(visualization_path, 'wb') as f:
            f.write(buf.getvalue())

        plt.close()

        return {visualization_path: VisualizationType.IMAGE}
