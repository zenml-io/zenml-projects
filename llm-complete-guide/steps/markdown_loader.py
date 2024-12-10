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

import os
import tempfile
from typing import Annotated

import polars as pl
from constants import FILES_TO_IGNORE
from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_markdown_files(
    git_repo_url: str = "https://github.com/zenml-io/zenml",
    subfolder: str = "docs/book",
) -> Annotated[pl.DataFrame, "markdown_files"]:
    """Loads markdown files from a given git repository URL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # clone the repository (top level of main branch only) into a temporary directory
        repo_name = git_repo_url.split("/")[-1]
        temp_repo_path = os.path.join(temp_dir, repo_name)
        os.system(
            f"git clone --single-branch --branch main --depth 1 {git_repo_url} {temp_repo_path}"
        )

        # get all markdown files from the subfolder
        markdown_files = []
        subfolder_path = os.path.join(temp_repo_path, subfolder)
        if os.path.exists(subfolder_path):
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file.endswith(".md") and file not in FILES_TO_IGNORE:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r") as f:
                            file_contents = f.read()
                            markdown_files.append(
                                (
                                    file_path.replace(
                                        temp_repo_path + "/", ""
                                    ),
                                    file_contents,
                                )
                            )
        else:
            raise FileNotFoundError(
                f"Subfolder '{subfolder}' not found in the cloned repository."
            )

    log_metadata(
        artifact_name="markdown_files",
        infer_artifact=True,
        metadata={
            "num_markdown_files": len(markdown_files),
            "columns": "filename, page_content",
            "repo_url": git_repo_url,
            "subfolder": subfolder,
            "repo_temp_dir": temp_dir,
        },
    )
    # return a polars DataFrame that includes columns for the filename and the
    # content of the markdown file
    return pl.DataFrame(
        {
            "filename": [f[0] for f in markdown_files],
            "page_content": [f[1] for f in markdown_files],
        }
    )
