import argparse

from constants import CLAUDE_3_HAIKU_MODEL_ID
from pipelines.bedrock_basic_inference import bedrock_basic_inference
from pipelines.bedrock_rag import bedrock_rag
from zenml.logger import get_logger
from zenml.client import Client

logger = get_logger(__name__)

sample_prompt = "What is the capital of France and what is the capital of Germany?"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run bedrock basic inference or RAG."
    )
    parser.add_argument(
        "--model_id",
        default=CLAUDE_3_HAIKU_MODEL_ID,
        help="The model ID to use for inference (default: CLAUDE_3_HAIKU_MODEL_ID)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["inference", "rag"],
        default="inference",
        help="The pipeline to run (default: inference)",
    )
    args = parser.parse_args()

    if args.pipeline == "inference":
        print("Asking model: ", sample_prompt)
        bedrock_basic_inference(
            model_id=args.model_id,
            prompt=sample_prompt,
        )
        zc = Client()
        output = zc.get_pipeline("bedrock_basic_inference").last_successful_run.steps['basic_inference'].outputs['output'].load()
        print(f"Answer: '{output}'")
    elif args.pipeline == "rag":
        bedrock_rag()
