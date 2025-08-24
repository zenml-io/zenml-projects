"""
Fetch source code workspace step.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Dict, Tuple

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def fetch_source(
    source_spec: Dict[str, str],
) -> Tuple[Annotated[Path, "workspace_dir"], Annotated[str, "commit_sha"]]:
    """
    Fetch and materialize workspace from git repository.

    Args:
        source_spec: Source specification from select_input step

    Returns:
        Tuple of workspace directory path and commit SHA
    """
    repo_url = source_spec["repo_url"]
    ref = source_spec["ref"]

    logger.info(f"Fetching source from {repo_url}@{ref}")

    # Create temporary workspace
    workspace_dir = tempfile.mkdtemp(prefix="qualityflow_workspace_")
    workspace_path = Path(workspace_dir)

    try:
        # Clone repository
        logger.info(f"Cloning {repo_url} to {workspace_dir}")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ref,
                repo_url,
                workspace_dir,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Get commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_sha = result.stdout.strip()

        logger.info(
            f"Workspace ready at {workspace_dir}, commit: {commit_sha}"
        )

        return Path(workspace_dir), commit_sha

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fetch source: {e}")
        raise RuntimeError(f"Git operation failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error fetching source: {e}")
        # Clean up on error
        import shutil

        shutil.rmtree(workspace_dir, ignore_errors=True)
        raise
