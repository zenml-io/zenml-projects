# ZenML pipelines for the email search agent
from pipelines.data_preparation import data_preparation_pipeline
from pipelines.evaluation import evaluation_pipeline
from pipelines.inference import inference_pipeline
from pipelines.training import training_pipeline

__all__ = [
    "data_preparation_pipeline",
    "training_pipeline",
    "evaluation_pipeline",
    "inference_pipeline",
]
