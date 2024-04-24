"""
Fine-Tune StarCoder on code/text dataset

Based off Sayak Paul (https://github.com/sayakpaul) and Sourab Mangrulkar (https://github.com/pacman100) codebase: https://github.com/pacman100/DHS-LLM-Workshop/tree/main/
All credit to them for their amazing work!
"""

import os
import subprocess
from multiprocessing import Pool
from typing import List

from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

ORG = "zenml-io"
MIRROR_DIRECTORY = "cloned_public_repos"


def mirror_repository(repository):
    """Locally clones a repository."""
    repository_url = f"https://github.com/{ORG}/{repository}.git"
    repository_path = os.path.join(MIRROR_DIRECTORY, repository)

    # Clone the repository
    subprocess.run(["git", "clone", repository_url, repository_path])


@step
def mirror_repositories(
    repositories: List[str],
) -> Annotated[str, "mirror_directory"]:
    """Locally clones a list of repositories.

    Args:
        repositories (List[str]): Names of the repositories to clone.

    Raises:
        ValueError: If the GH_ACCESS_TOKEN environment variable is not set.
    """
    # Create the mirror directory if it doesn't exist
    if not os.path.exists(MIRROR_DIRECTORY):
        os.makedirs(MIRROR_DIRECTORY)

    # Get the GitHub access token
    gh_access_token = None
    gh_access_token = os.getenv("GH_ACCESS_TOKEN", None)
    client = Client()

    # Try to get the access token from the ZenML client
    try:
        gh_access_token = client.get_secret("GH_ACCESS_TOKEN").secret_values[
            "token"
        ]
    except KeyError:
        pass

    # Raise an error if the access token is not found
    if gh_access_token is None:
        raise ValueError(
            "Please set the GH_ACCESS_TOKEN environment variable."
        )

    # Get the list of repositories in the organization
    print(f"Total repositories found: {len(repositories)}.")

    # Mirror repositories using multiprocessing
    print("Cloning repositories.")
    with Pool() as pool:
        pool.map(mirror_repository, repositories)

    return MIRROR_DIRECTORY
