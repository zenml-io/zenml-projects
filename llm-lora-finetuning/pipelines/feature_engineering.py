from steps.feature_engineering import feature_engineering
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(settings={"docker": DockerSettings(requirements="requirements.txt")})
def feature_engineering_pipeline(model_repo: str, dataset_name: str) -> None:
    feature_engineering(model_repo=model_repo, dataset_name=dataset_name)
