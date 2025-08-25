"""
Fetch source code workspace step.

This module provides functionality to clone Git repositories and prepare
workspaces for code analysis and test generation.
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
    Fetch and materialize workspace from git repository or use local examples.

    Args:
        source_spec: Source specification from select_input step

    Returns:
        Tuple of workspace directory path and commit SHA
    """
    repo_url = source_spec["repo_url"]
    ref = source_spec["ref"]

    # Handle local examples case
    if repo_url == "local":
        logger.info("Using local QualityFlow examples")
        try:
            # Get the project root (QualityFlow directory)
            current_file = Path(__file__).resolve()
            project_root = (
                current_file.parent.parent
            )  # Go up from steps/ to project root

            # Create temporary workspace and copy examples
            workspace_dir = tempfile.mkdtemp(
                prefix="qualityflow_local_workspace_"
            )
            workspace_path = Path(workspace_dir)

            # Copy examples directory to the temporary workspace
            import shutil

            examples_src = project_root / "examples"
            examples_dest = workspace_path / "examples"

            if examples_src.exists():
                shutil.copytree(examples_src, examples_dest)
                logger.info(
                    f"Copied examples from {examples_src} to {examples_dest}"
                )
            else:
                logger.warning(
                    f"Examples directory not found at {examples_src}"
                )

            commit_sha = "local-examples"
            logger.info(f"Local workspace ready at {workspace_path}")
            return workspace_path, commit_sha

        except Exception as e:
            logger.error(f"Failed to set up local workspace: {e}")
            # Fallback to current working directory
            workspace_dir = tempfile.mkdtemp(
                prefix="qualityflow_fallback_workspace_"
            )
            return Path(workspace_dir), "local-fallback"

    logger.info(f"Fetching source from {repo_url}@{ref}")

    # Create temporary workspace for remote repositories
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
