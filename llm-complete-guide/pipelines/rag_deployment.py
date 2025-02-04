from steps.rag_deployment import gradio_rag_deployment
from zenml import pipeline

from typing import Optional


@pipeline(enable_cache=False)
def rag_deployment(mlflow_experiment_name: Optional[str] = None):
    gradio_rag_deployment(mlflow_experiment_name=mlflow_experiment_name)
