from constants import CLAUDE_3_HAIKU_MODEL_ID
from pipelines.bedrock_basic_inference import bedrock_basic_inference
from zenml.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    bedrock_basic_inference(
        model_id=CLAUDE_3_HAIKU_MODEL_ID,
        prompt="What is the capital of France and what is the capital of Germany?",
    )
