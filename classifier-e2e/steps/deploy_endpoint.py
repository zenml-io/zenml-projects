from datetime import datetime

import sagemaker
from sagemaker import Predictor
from sagemaker.image_uris import retrieve
from typing_extensions import Annotated
from utils.aws import get_aws_config
from utils.sagemaker_materializer import SagemakerPredictorMaterializer
from zenml import ArtifactConfig, get_step_context, log_artifact_metadata, step


@step(
    enable_cache=False,
    output_materializers=[SagemakerPredictorMaterializer],
)
def deploy_endpoint() -> (
    Annotated[
        Predictor,
        ArtifactConfig(name="sagemaker_endpoint", is_deployment_artifact=True),
    ]
):
    role, session, region = get_aws_config()

    model = get_step_context().model._get_model_version()
    if "sgd" in {t.name for t in model.tags}:
        image_uri = retrieve(region=region, framework="sklearn", version="1.0-1")
        entry_point = "utils/sklearn_inference.py"
    else:
        image_uri = retrieve(region=region, framework="xgboost", version="1.5-1")
        entry_point = None

    model_data = f'{model.get_artifact("breast_cancer_classifier").uri}/model.tar.gz'

    endpoint_name = (
        f'breast-cancer-classifier-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}'
    )
    sagemaker.Model(
        image_uri=image_uri,
        model_data=model_data,
        sagemaker_session=session,
        role=role,
        entry_point=entry_point,
    ).deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name,
    )

    log_artifact_metadata(
        {
            "endpoint_name": endpoint_name,
            "image_uri": image_uri,
            "role_arn": role,
        }
    )

    return Predictor(endpoint_name=endpoint_name)
