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
#

from typing import Optional

import click
from constants import OPENAI_MODEL
from pipelines import (
    llm_basic_rag,
)
from utils.llm_utils import process_input_with_retrieval
from zenml.logger import get_logger

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
def main(
    rag: bool = False,
    query: Optional[str] = None,
    model: str = OPENAI_MODEL,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
        rag (bool): If `True`, the basic RAG pipeline will be run.
        query (Optional[str]): If provided, the RAG model will be queried with this string.
        model (str): The model to use for the completion. Default is OPENAI_MODEL.
        no_cache (bool): If `True`, cache will be disabled.

    """
    pipeline_args = {"enable_cache": not no_cache}

    if query:
        response = process_input_with_retrieval(query, model=model)
        print(response)

    if rag:
        llm_basic_rag.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
