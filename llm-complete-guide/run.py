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

import os
from typing import Optional

import click
from zenml.logger import get_logger

from pipelines import (
    llm_basic_rag,
)

logger = get_logger(__name__)


@click.command(
    help="""
ZenML LLM Complete Guide project CLI v0.1.0.

Run the ZenML LLM RAG complete guide project pipelines.

Examples:

  \b
  # Run the feature feature engineering pipeline
    python run.py --feature-pipeline
"""
)
@click.option(
    "--basic-rag",
    "basic_rag",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
def main(
    basic_rag: bool = False,
    no_cache: bool = False,
):
    """Main entry point for the pipeline execution.

    Args:
        no_cache: If `True` cache will be disabled.
    """
    pipeline_args = {"enable_cache": not no_cache}

    if basic_rag:
        llm_basic_rag.with_options(**pipeline_args)()


if __name__ == "__main__":
    main()
