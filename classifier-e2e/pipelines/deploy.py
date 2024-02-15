from zenml import pipeline, get_pipeline_context
from steps import data_loader, inference_preprocessor
import random
from steps import deploy_endpoint, predict_on_endpoint, shutdown_endpoint


@pipeline
def deploy(shutdown_endpoint_after_predicting: bool = True):
    # Get the preprocess pipeline artifact associated with this version
    preprocess_pipeline = get_pipeline_context().model.get_artifact(
        "preprocess_pipeline"
    )

    df_inference = data_loader(
        random_state=random.randint(0, 1000), is_inference=True
    )
    df_inference = inference_preprocessor(
        dataset_inf=df_inference,
        preprocess_pipeline=preprocess_pipeline,
        target="target",
    )
    endpoint_name = deploy_endpoint()
    predict_on_endpoint(endpoint_name, df_inference)
    if shutdown_endpoint_after_predicting:
        shutdown_endpoint(endpoint_name, after=["predict_on_endpoint"])
