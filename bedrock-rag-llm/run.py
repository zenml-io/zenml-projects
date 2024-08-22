from constants import CLAUDE_3_HAIKU_MODEL_ID
from steps.basic_inference import basic_inference
from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@pipeline
def bedrock_basic_inference(model_id: str, prompt: str):
    basic_inference(model_id, prompt)


if __name__ == "__main__":
    bedrock_basic_inference(
        model_id=CLAUDE_3_HAIKU_MODEL_ID,
        prompt="What is the capital of France and what is the capital of Germany?",
    )
