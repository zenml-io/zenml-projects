# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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

import os
import io
from typing import Tuple, Dict, Any
from typing_extensions import Annotated

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from zenml import pipeline, step, Model, ArtifactConfig
from zenml.logger import get_logger
from zenml import log_artifact_metadata, log_model_metadata
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.config import DockerSettings
from zenml.materializers.base_materializer import BaseMaterializer

logger = get_logger(__name__)


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


def safe_metadata(data: Any) -> Dict[str, Any]:
    """Create metadata dict with only supported types."""
    metadata = {"shape": data.shape}
    if isinstance(data, pd.DataFrame):
        metadata["columns"] = list(data.columns)
    return metadata


@step
def load_data() -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Load the iris dataset and split into train and test sets."""
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, data in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
        log_artifact_metadata(
            artifact_name=name,
            metadata={"dataset_info": safe_metadata(data)}
        )

    return X_train, X_test, y_train, y_test


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Annotated[SVC, ArtifactConfig(name="model", artifact_type=ArtifactType.MODEL)]:
    """Train an SVM classifier."""
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)

    log_model_metadata(
        metadata={
            "training_metrics": {
                "train_accuracy": float(train_accuracy),
            },
            "model_info": {
                "model_type": type(model).__name__,
                "kernel": model.kernel,
            }
        }
    )

    log_artifact_metadata(
        artifact_name="model",
        metadata={
            "model_details": {
                "type": type(model).__name__,
                "kernel": model.kernel,
                "n_support": model.n_support_.tolist(),
            }
        }
    )

    return model


@step
def evaluate_model(
    model: SVC,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[np.ndarray, "predictions"],
    Annotated[np.ndarray, "probabilities"]
]:
    """Evaluate the model and make predictions."""
    test_accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    log_model_metadata(
        metadata={
            "evaluation_metrics": {
                "test_accuracy": float(test_accuracy),
            }
        }
    )

    log_artifact_metadata(
        artifact_name="predictions",
        metadata={
            "prediction_info": {
                "shape": predictions.shape,
                "unique_values": np.unique(predictions).tolist()
            }
        }
    )

    log_artifact_metadata(
        artifact_name="probabilities",
        metadata={
            "probability_info": {
                "shape": probabilities.shape,
                "min": float(np.min(probabilities)),
                "max": float(np.max(probabilities))
            }
        }
    )

    return predictions, probabilities


@step
def explain_model(
    model: SVC,
    X_train: pd.DataFrame
) -> Annotated[SHAPVisualization, "shap_visualization"]:
    """Generate SHAP values for model explainability and create a visualization."""
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
    shap_values = explainer.shap_values(X_train.iloc[:100])

    log_artifact_metadata(
        artifact_name="shap_values",
        metadata={
            "shap_info": {
                "shape": [arr.shape for arr in shap_values],
                "n_classes": len(shap_values),
                "n_features": shap_values[0].shape[1],
            }
        }
    )

    return SHAPVisualization(shap_values, X_train.columns)


@step
def detect_data_drift(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Annotated[Dict[str, float], "drift_metrics"]:
    """Detect data drift between training and test sets."""
    drift_metrics = {}
    for column in X_train.columns:
        _, p_value = ks_2samp(X_train[column], X_test[column])
        drift_metrics[column] = p_value

    log_artifact_metadata(
        artifact_name="drift_metrics",
        metadata={
            "drift_summary": {
                "high_drift_features": [col for col, p in drift_metrics.items() if p < 0.05]
            }
        }
    )

    return drift_metrics


@pipeline(
    enable_cache=False,
    settings={"docker": DockerSettings(requirements="requirements.txt")},
    model=Model(name="high_risk_classification")
)
def iris_classification_pipeline():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    explain_model(model, X_train)
    drift_metrics = detect_data_drift(X_train, X_test)


if __name__ == "__main__":
    iris_classification_pipeline()
