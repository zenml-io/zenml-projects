from steps.rag_deployment import gradio_rag_deployment
from zenml import pipeline


@pipeline(enable_cache=False)
def rag_deployment():
    gradio_rag_deployment()
