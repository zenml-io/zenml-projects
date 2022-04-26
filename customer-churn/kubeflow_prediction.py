import pandas as pd
from sklearn.base import ClassifierMixin
from zenml.pipelines import pipeline
from zenml.repository import Repository
from zenml.steps import Output, StepContext, step


@step
def prediction(context: StepContext) -> ClassifierMixin:
    pipeline_runs = context.metadata_store.get_pipeline("training_pipeline").runs
    for run in pipeline_runs:
        trainer_step = run.get_step("trainer")
        model = trainer_step.output
    return model


@pipeline(requirements_file="kubeflow_requirements.txt")
def prediction_pipeline(prediction):
    prediction = prediction()


if __name__ == "__main__":
    pred = prediction_pipeline(prediction())
    pred.run()
