from typing_extensions import Annotated

import sagemaker
from sagemaker.image_uris import retrieve

from zenml import step, get_step_context
from datetime import datetime

from utils.aws import get_aws_config


@step(enable_cache=False)
def deploy_endpoint() -> Annotated[str, "sagemaker_endpoint_name"]:
    role, session, region = get_aws_config()

    model = get_step_context().model._get_model_version()
    if "sgd" in {t.name for t in model.tags}:
        image_uri = retrieve(
            region=region, framework="sklearn", version="1.0-1"
        )
        entry_point = "utils/sklearn_inference.py"
    else:
        image_uri = retrieve(
            region=region, framework="xgboost", version="1.5-1"
        )
        entry_point = None

    model_data = (
        f'{model.get_artifact("breast_cancer_classifier").uri}/model.tar.gz'
    )

    endpoint_name = f'breast-cancer-classifier-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")}'
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
    return endpoint_name
