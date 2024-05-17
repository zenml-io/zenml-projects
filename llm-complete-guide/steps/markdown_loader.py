import os
import tempfile

import polars as pl
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def load_markdown_files(
    git_repo_url: str = "https://github.com/zenml-io/zenml",
    subfolder: str = "docs",
) -> pl.DataFrame:
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
                    if file.endswith(".md"):
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

    # return a polars DataFrame that includes columns for the filename and the
    # content of the markdown file
    return pl.DataFrame(
        {
            "filename": [f[0] for f in markdown_files],
            "page_content": [f[1] for f in markdown_files],
        }
    )
