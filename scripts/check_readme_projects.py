#!/usr/bin/env python3
"""
CI check to ensure all project directories are listed in the README table.

This script checks for project directories (excluding certain directories like _assets,
scripts, etc.) and ensures that they are all referenced in the project table in the
main README.md file.

Projects can be exempted from this check by adding them to the exempt_projects set
in the get_project_directories function. This is useful for work-in-progress projects,
internal projects, or projects that are not meant to be public-facing.
"""

import os
import re
import sys
from pathlib import Path


def get_project_directories(repo_root):
    """Get a list of project directories from the repository.

    Args:
        repo_root: The root directory of the repository.

    Returns:
        List of project directory names.
    """
    # Directories to exclude (infrastructure, config, assets, etc.)
    exclude_dirs = {
        "_assets",
        "scripts",
        "assets",
        ".git",
        "__pycache__",
        ".github",
        "wandb",
    }

    # Projects to exempt from README table requirement
    # Add directories here that don't need to be in the README table
    exempt_projects = {
        # Work-in-progress or internal projects
        "finscan",
        "sonicscribe",
    }

    project_dirs = []

    for item in os.listdir(repo_root):
        item_path = os.path.join(repo_root, item)

        # Check if the item is a directory and not in the exclude or exempt lists
        if (
            os.path.isdir(item_path)
            and item not in exclude_dirs
            and item not in exempt_projects
            and not item.startswith(".")
        ):
            # Skip directories that are Python package-related but not actual projects
            if not item.startswith("__") and item != "venv" and item != "env":
                project_dirs.append(item)

    return project_dirs


def get_readme_projects(readme_path):
    """Extract project directories listed in the README table.

    Args:
        readme_path: Path to the README.md file.

    Returns:
        List of project directory names referenced in the README.
    """
    with open(readme_path, "r") as f:
        readme_content = f.read()

    # Find the project table
    table_pattern = r"\| Project\s+\| Domain.*?\n(.*?)(?:\n\n|\n#)"
    table_match = re.search(table_pattern, readme_content, re.DOTALL)

    if not table_match:
        print("Error: Could not find project table in README.md")
        return []

    table_content = table_match.group(1)

    # Extract project links from the table
    # The pattern looks for Markdown links like [ProjectName](directory)
    link_pattern = r"\[.*?\]\((.*?)\)"
    project_links = re.findall(link_pattern, table_content)

    # Convert links to directory names
    readme_projects = []
    for link in project_links:
        # Remove trailing slash if present
        if link.endswith("/"):
            link = link[:-1]
        readme_projects.append(link)

    return readme_projects


def main():
    """Main function to run the check."""
    # Get the repository root
    repo_root = Path(__file__).parent.parent.absolute()

    # Get project directories from the repository (already excludes exempted projects)
    project_dirs = get_project_directories(repo_root)

    # Get projects listed in the README
    readme_path = os.path.join(repo_root, "README.md")
    readme_projects = get_readme_projects(readme_path)

    # Find missing projects
    missing_projects = set(project_dirs) - set(readme_projects)

    if missing_projects:
        print(
            "Error: The following project directories are not listed in the README table:"
        )
        for project in sorted(missing_projects):
            print(f"  - {project}")
        print(
            "\nTo exempt a project from this check, add it to the exempt_projects set in this script."
        )
        return 1

    print(
        "Success: All required project directories are listed in the README table."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
