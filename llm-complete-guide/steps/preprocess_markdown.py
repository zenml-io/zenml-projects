#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import polars as pl
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def preprocess_markdown_texts(markdown_texts: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses markdown texts.

    This function performs the following preprocessing steps on the markdown texts:
    1. Removes image links in the format ![alt text](image url)
    2. Removes bash/shell code blocks enclosed in triple backticks (```)
    3. Removes the boilerplate section at the top of the document enclosed in triple hyphens (---)

    Args:
        markdown_texts (pl.DataFrame): A DataFrame containing the markdown texts to preprocess.
            The DataFrame should have a column named "page_content" containing the markdown text.

    Returns:
        pl.DataFrame: The preprocessed DataFrame with the same structure as the input DataFrame.
    """
    markdown_texts = markdown_texts.with_columns(
        pl.col("page_content")
        .str.replace_all(r"!\[.*\]\(.*\)", "")
        .alias("page_content")
    )
    markdown_texts = markdown_texts.with_columns(
        pl.col("page_content")
        .str.replace_all(r"```(bash|shell)(.|\n)*?```", "")
        .alias("page_content")
    )
    markdown_texts = markdown_texts.with_columns(
        pl.col("page_content")
        .str.replace_all(r"---\n.*?\n---\n", "")
        .alias("page_content")
    )

    return markdown_texts
