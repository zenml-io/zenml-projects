import argparse

from constants import CLAUDE_3_HAIKU_MODEL_ID
from pipelines.bedrock_basic_inference import bedrock_basic_inference
from zenml.logger import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bedrock basic inference."
    )
    parser.add_argument(
        "--model_id",
        default=CLAUDE_3_HAIKU_MODEL_ID,
        help="The model ID to use for inference (default: CLAUDE_3_HAIKU_MODEL_ID)",
    )
    args = parser.parse_args()

    bedrock_basic_inference(
        model_id=args.model_id,
        prompt="What is the capital of France and what is the capital of Germany?",
    )
