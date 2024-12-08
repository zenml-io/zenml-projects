from steps.bento_builder import bento_builder
from steps.bento_deployment import bento_deployment
from zenml import pipeline


@pipeline(enable_cache=False)
def local_deployment():
    bento = bento_builder()
    bento_deployment(bento)
    
    #vllm_model_deployer_step()
