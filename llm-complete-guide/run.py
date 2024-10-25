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
import os
import warnings

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
)
from structures import Document
from zenml.materializers.materializer_registry import materializer_registry

logger = get_logger(__name__)


@click.command(
    help="""
ZenML LLM Complete Guide project CLI v0.1.0.

Run the ZenML LLM RAG complete guide project pipelines.
"""
)
@click.option(
    "--rag",
    "rag",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--evaluation",
    "evaluation",
    is_flag=True,
    default=False,
    help="Whether to run the evaluation pipeline.",
)
@click.option(
    "--query",
    "query",
    type=str,
    required=False,
    help="Query the RAG model.",
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
    "--no-cache",
    "no_cache",
    is_flag=True,
    default=False,
    help="Disable cache.",
)
@click.option(
    "--synthetic",
    "synthetic",
    is_flag=True,
    default=False,
    help="Run the synthetic data pipeline.",
)
@click.option(
    "--embeddings",
    "embeddings",
    is_flag=True,
    default=False,
    help="Fine-tunes embeddings.",
)
@click.option(
    "--argilla",
    "argilla",
    is_flag=True,
    default=False,
    help="Uses Argilla annotations.",
)
@click.option(
    "--reranked",
    "reranked",
    is_flag=True,
    default=False,
    help="Whether to use the reranker.",
)
@click.option(
    "--chunks",
    "chunks",
    is_flag=True,
    default=False,
    help="Generate chunks for Hugging Face dataset",
)
def main(
    rag: bool = False,
    evaluation: bool = False,
    query: Optional[str] = None,
    model: str = OPENAI_MODEL,
    no_cache: bool = False,
    synthetic: bool = False,
    embeddings: bool = False,
    argilla: bool = False,
    reranked: bool = False,
    chunks: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
        rag (bool): If `True`, the basic RAG pipeline will be run.
        evaluation (bool): If `True`, the evaluation pipeline will be run.
        query (Optional[str]): If provided, the RAG model will be queried with this string.
        model (str): The model to use for the completion. Default is OPENAI_MODEL.
        no_cache (bool): If `True`, cache will be disabled.
        synthetic (bool): If `True`, the synthetic data pipeline will be run.
        local (bool): If `True`, the local LLM via Ollama will be used.
        embeddings (bool): If `True`, the embeddings will be fine-tuned.
        argilla (bool): If `True`, the Argilla annotations will be used.
        chunks (bool): If `True`, the chunks pipeline will be run.
        reranked (bool): If `True`, rerankers will be used
    """
    pipeline_args = {"enable_cache": not no_cache}
    embeddings_finetune_args = {
        "enable_cache": not no_cache,
        "steps": {
            "prepare_load_data": {
                "parameters": {"use_argilla_annotations": argilla}
            }
        },
    }

    if query:
        response = process_input_with_retrieval(
            query, model=model, use_reranking=reranked
        )

        # print rich markdown to the console
        console = Console()
        md = Markdown(response)
        console.print(md)

    print(f"Running Pipeline with pipeline args: {pipeline_args}")
    if rag:
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "rag.yaml"
        )
        llm_basic_rag.with_options(config_path=config_path, **pipeline_args)()
    if evaluation:
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "rag_eval.yaml"
        )
        pipeline_args["enable_cache"] = False
        llm_eval.with_options(config_path=config_path)()
    if synthetic:
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "synthetic.yaml"
        )
        generate_synthetic_data.with_options(config_path=config_path, **pipeline_args)()
    if embeddings:
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "configs", "embeddings.yaml"
        )
        finetune_embeddings.with_options(config_path=config_path, **embeddings_finetune_args)()
    if chunks:
        generate_chunk_questions.with_options(**pipeline_args)()


if __name__ == "__main__":
    # use custom materializer for documents
    # register early
    materializer_registry.register_materializer_type(
        Document, DocumentMaterializer
    )
    main()
