import argparse

from constants import CLAUDE_3_HAIKU_MODEL_ID
from pipelines.bedrock_basic_inference import bedrock_basic_inference
from pipelines.bedrock_custom_model_finetuning import (
    bedrock_custom_model_finetuning,
)
from pipelines.bedrock_rag import bedrock_rag
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)

sample_prompt = (
    "What is the capital of France and what is the capital of Germany?"
)

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
        choices=["inference", "rag", "finetune"],
        default="inference",
        help="The pipeline to run (default: inference)",
    )
    parser.add_argument(
        "--provision",
        default=False,
        help="Whether to provision the RAG stack on Bedrock (default: False)",
    )
    parser.add_argument(
        "--query",
        default=sample_prompt,
        help="The query to use for RAG (default: sample_prompt)",
    )
    parser.add_argument(
        "--dataset",
        default="data",
        help="The directory to use for the dataset (default: data)",
    )
    args = parser.parse_args()

    if args.pipeline == "inference":
        print("Asking model: ", args.query)
        bedrock_basic_inference(
            model_id=args.model_id,
            prompt=args.query,
        )
        zc = Client()
        output = (
            zc.get_pipeline("bedrock_basic_inference")
            .last_successful_run.steps["basic_inference"]
            .outputs["output"]
            .load()
        )
        print(f"Answer: '{output}'")
    elif args.pipeline == "rag":
        if args.provision:
            # sets up permissions
            # creates knowledge base + ingests data
            bedrock_rag(provision=True)
        else:
            # inference on your bedrock knowledge base
            bedrock_rag(provision=False, query=args.query)
    elif args.pipeline == "finetune":
        bedrock_custom_model_finetuning(dataset_dir=args.dataset)
