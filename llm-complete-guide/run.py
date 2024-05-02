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
import logging

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
from pipelines import generate_chunk_questions, llm_basic_rag, llm_eval
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
    "--local",
    "local",
    is_flag=True,
    default=False,
    help="Uses a local LLM via Ollama.",
)
def main(
    rag: bool = False,
    evaluation: bool = False,
    query: Optional[str] = None,
    model: str = OPENAI_MODEL,
    no_cache: bool = False,
    synthetic: bool = False,
    local: bool = False,
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
    """
    pipeline_args = {"enable_cache": not no_cache}

    if query:
        response = process_input_with_retrieval(query, model=model)
        print(response)

    if rag:
        llm_basic_rag.with_options(**pipeline_args)()
    if evaluation:
        llm_eval.with_options(**pipeline_args)()
    if synthetic:
        generate_chunk_questions.with_options(**pipeline_args)()


if __name__ == "__main__":
    # use custom materializer for documents
    # register early
    materializer_registry.register_materializer_type(
        Document, DocumentMaterializer
    )
    main()
