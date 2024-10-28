from zenml import pipeline
from steps.rag_deployment import gradio_rag_deployment
@pipeline(enable_cache=False)
def rag_deployment():
    gradio_rag_deployment()
