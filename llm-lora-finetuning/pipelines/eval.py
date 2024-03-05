from typing import Optional

from steps.eval import eval
from zenml import pipeline
from zenml.config import DockerSettings


@pipeline(settings={"docker": DockerSettings(requirements="requirements.txt")})
def eval_pipeline(model_repo: str, adapter_repo: Optional[str] = None) -> None:
    eval(model_repo=model_repo, adapter_repo=adapter_repo)
