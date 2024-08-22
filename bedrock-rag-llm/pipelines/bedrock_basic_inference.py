from steps.basic_inference import basic_inference
from zenml import pipeline


@pipeline
def bedrock_basic_inference(model_id: str, prompt: str):
    basic_inference(model_id, prompt)
