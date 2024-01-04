# {% include 'template/license_header' %}

from typing import Optional, List

from steps import (
    deploy_to_huggingface,
)
from zenml import get_pipeline_context, pipeline
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)


@pipeline
def breast_cancer_deployment_pipeline(
    repo_name: Optional[str] = "zenml_breast_cancer_classifier",
):
    """
    Model deployment pipeline.

    This pipelines deploys latest model on mlflow registry that matches
    the given stage, to one of the supported deployment targets.

    Args:
        labels: List of labels for the model.
        title: Title for the model.
        description: Description for the model.
        model_name_or_path: Name or path of the model.
        tokenizer_name_or_path: Name or path of the tokenizer.
        interpretation: Interpretation for the model.
        example: Example for the model.
        repo_name: Name of the repository to deploy to HuggingFace Hub.
    """
    ########## Deploy to HuggingFace ##########
    deploy_to_huggingface(
        repo_name=repo_name,
        after=["save_model_to_deploy"],
    )
