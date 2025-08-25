"""
Select input source specification step.
"""

from typing import Annotated, Dict

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def select_input(
    repo_url: str = "https://github.com/psf/requests.git",
    ref: str = "main",
    target_glob: str = "src/**/*.py",
) -> Annotated[Dict[str, str], "source_spec"]:
    """
    Resolve source specification for test generation.

    Args:
        repo_url: Repository URL to analyze
        ref: Git reference (branch, tag, commit)
        target_glob: Glob pattern for target files

    Returns:
        Source specification dictionary
    """
    logger.info(f"Selecting input source: {repo_url}@{ref}")

    spec = {
        "repo_url": repo_url,
        "ref": ref,
        "target_glob": target_glob,
    }

    logger.info(f"Source spec: {spec}")
    return spec
