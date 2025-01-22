from sklearn.base import ClassifierMixin
from zenml import get_step_context, log_artifact_metadata
import shap
import pandas as pd
from typing import Annotated
from zenml.steps import step
from .shap_visualization import SHAPVisualization

@step
def explain_model(
    X_train: pd.DataFrame
) -> Annotated[SHAPVisualization, "shap_visualization"]:
    """Generate SHAP values for model explainability and create a visualization."""
    model = get_step_context().model
    model_artifact: ClassifierMixin = model.load_artifact("model")
    
    explainer = shap.KernelExplainer(model_artifact.predict_proba, shap.sample(X_train, 100))
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