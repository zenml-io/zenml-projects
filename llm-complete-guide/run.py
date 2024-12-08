# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from pathlib import Path

# Suppress the specific FutureWarning from huggingface_hub
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="huggingface_hub.file_download"
)

import logging

from rich.console import Console
from rich.markdown import Markdown
from utils.llm_utils import process_input_with_retrieval
from zenml.logger import get_logger

# Next, configure the loggers right after the imports
logging.getLogger("pytorch").setLevel(logging.CRITICAL)
logging.getLogger("sentence-transformers").setLevel(logging.CRITICAL)
logging.getLogger("rerankers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)  # Root logger configuration

# Continue with the rest of your imports and code
from typing import Optional

import click
from constants import OPENAI_MODEL
from materializers.document_materializer import DocumentMaterializer
from pipelines import (
    finetune_embeddings,
    generate_chunk_questions,
    generate_synthetic_data,
    llm_basic_rag,
    llm_eval,
    rag_deployment,
    llm_index_and_evaluate,
    local_deployment,
)
from structures import Document
from zenml.materializers.materializer_registry import materializer_registry
from zenml import Model

logger = get_logger(__name__)


@click.command(
    help="""
ZenML LLM Complete Guide project CLI v0.1.0.

Run the ZenML LLM RAG complete guide project pipelines.
"""
)
@click.argument(
    "pipeline",
    type=click.Choice(
        [
            "rag",
            "deploy",
            "evaluation",
            "query",
            "synthetic",
            "embeddings",
            "chunks",
            "basic_rag",
        ]
    ),
    required=True,
)
@click.option(
    "--model",
    "model",
    type=click.Choice(
        [
            "gpt4",
            "gpt35",
            "claude3",
            "claudehaiku",
        ]
    ),
    required=False,
    default="gpt4",
    help="The model to use for the completion.",
)
@click.option(
    "--query-text",
    "query_text",
    required=False,
    default=None,
    help="The query text to use for the completion.",
)
@click.option(
    "--zenml-model-name",
    "zenml_model_name",
    default="zenml-docs-qa-chatbot",
    required=False,
    help="The name of the ZenML model to use.",
)
@click.option(
    "--zenml-model-version",
    "zenml_model_version",
    required=False,
    default=None,
    help="The name of the ZenML model version to use.",
)
@click.option(
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable cache.",
)
@click.option(
    "--argilla",
    "use_argilla",
    is_flag=True,
    default=False,
    help="Uses Argilla annotations.",
)
@click.option(
    "--reranked",
    "use_reranker",
    is_flag=True,
    default=False,
    help="Whether to use the reranker.",
)
@click.option(
    "--config",
    "config",
    default=None,
    help="Path to config",
)
def main(
    pipeline: str,
    query_text: Optional[str] = None,
    model: str = OPENAI_MODEL,
    zenml_model_name: str = "zenml-docs-qa-chatbot",
    zenml_model_version: str = None,
    no_cache: bool = False,
    use_argilla: bool = False,
    use_reranker: bool = False,
    config: Optional[str] = None,
):
    """Main entry point for the pipeline execution.

    Args:
        pipeline (str): The pipeline to execute (rag, deploy, evaluation, etc.)
        query_text (Optional[str]): Query text when using 'query' command
        model (str): The model to use for the completion
        zenml_model_name (str): The name of the ZenML model to use
        zenml_model_version (str): The name of the ZenML model version to use
        no_cache (bool): If True, cache will be disabled
        use_argilla (bool): If True, Argilla an notations will be used
        use_reranker (bool): If True, rerankers will be used
        config (Optional[str]): Path to config file
    """
    pipeline_args = {"enable_cache": not no_cache}
    embeddings_finetune_args = {
        "enable_cache": not no_cache,
        "steps": {
            "prepare_load_data": {
                "parameters": {"use_argilla_annotations": use_argilla}
            }
        },
    }
    
    # Read the model version from a file in the root of the repo
    #  called "ZENML_VERSION.txt".    
    if zenml_model_version == "staging":
        postfix = "-rc0"
    elif zenml_model_version == "production":
        postfix = ""
    else:
        postfix = "-dev"

    if Path("ZENML_VERSION.txt").exists():
        with open("ZENML_VERSION.txt", "r") as file:
            zenml_model_version = file.read().strip()
            zenml_model_version += postfix
    else:
        raise RuntimeError(
            "No model version file found. Please create a file called ZENML_VERSION.txt in the root of the repo with the model version."
        )

    # Create ZenML model
    zenml_model = Model(
        name=zenml_model_name,
        version=zenml_model_version,
        license="Apache 2.0",
        description="RAG application for ZenML docs",
        tags=["rag", "finetuned", "chatbot"],
        limitations="Only works for ZenML documentation. Not generalizable to other domains. Entirely build with synthetic data. The data is also quite noisy on account of how the chunks were split.",
        trade_offs="Focused on a specific RAG retrieval use case. Not generalizable to other domains.",
        audience="ZenML users",
        use_cases="RAG retrieval",
    )

    # Handle config path
    config_path = None
    if config:
        config_path = Path(__file__).parent / "configs" / config

    # Set default config paths based on pipeline
    if not config_path:
        config_mapping = {
            "basic_rag": "dev/rag.yaml",
            "rag": "dev/rag.yaml",
            "evaluation": "dev/rag_eval.yaml",
            "synthetic": "dev/synthetic.yaml",
            "embeddings": "dev/embeddings.yaml",
        }
        if pipeline in config_mapping:
            config_path = (
                Path(__file__).parent / "configs" / config_mapping[pipeline]
            )

    # Execute query
    if pipeline == "query":
        if not query_text:
            raise click.UsageError(
                "--query-text is required when using 'query' command"
            )
        response = process_input_with_retrieval(
            query_text, model=model, use_reranking=use_reranker
        )
        console = Console()
        md = Markdown(response)
        console.print(md)
        return

    # Execute the appropriate pipeline
    if pipeline == "basic_rag":
        llm_basic_rag.with_options(
            model=zenml_model, config_path=config_path, **pipeline_args
        )()
        # Also deploy if config is provided
        if config:
            rag_deployment.with_options(
                config_path=config_path, **pipeline_args
            )()

    if pipeline == "rag":
        llm_index_and_evaluate.with_options(
            model=zenml_model, config_path=config_path, **pipeline_args
        )()

    elif pipeline == "deploy":
        #rag_deployment.with_options(model=zenml_model, **pipeline_args)()
        local_deployment.with_options(model=zenml_model, **pipeline_args)()

    elif pipeline == "evaluation":
        pipeline_args["enable_cache"] = False
        llm_eval.with_options(model=zenml_model, config_path=config_path)()

    elif pipeline == "synthetic":
        generate_synthetic_data.with_options(
            model=zenml_model, config_path=config_path, **pipeline_args
        )()

    elif pipeline == "embeddings":
        finetune_embeddings.with_options(
            model=zenml_model, config_path=config_path, **embeddings_finetune_args
        )()

    elif pipeline == "chunks":
        generate_chunk_questions.with_options(
            model=zenml_model, config_path=config_path, **pipeline_args
        )()


if __name__ == "__main__":
    # use custom materializer for documents
    # register early
    materializer_registry.register_materializer_type(
        Document, DocumentMaterializer
    )
    main()