#!/usr/bin/env python3

"""Script to generate Dockerfile.sandbox files for ZenML projects.

This ensures consistency across all project Docker images.
"""

import argparse
import os
import re
import sys
from pathlib import Path

DOCKERFILE_TEMPLATE = """# Sandbox base image
FROM safoinext/zenml-sandbox:latest

# Project metadata
LABEL project_name="{project_name}"
LABEL project_version="0.1.0"

# Install project-specific dependencies
RUN pip install --no-cache-dir \\
{dependencies}

# Set workspace directory
WORKDIR /workspace

# Clone only the project directory and reorganize
RUN git clone --depth 1 https://github.com/zenml-io/zenml-projects.git /tmp/zenml-projects && \\
    cp -r /tmp/zenml-projects/{project_name}/* /workspace/ && \\
    rm -rf /tmp/zenml-projects

# Create a template .env file for API keys
RUN echo "{api_vars}" > .env

# Create a .vscode directory and settings.json file
RUN mkdir -p /workspace/.vscode && \\
    echo '{{\\n'\\
    '  "workbench.colorTheme": "Default Dark Modern"\\n'\\
    '}}' > /workspace/.vscode/settings.json

{env_vars_block}
"""


def format_env_key(key):
    """Format environment variable placeholder text."""
    # Extract the service name from the key
    service = key.split("_")[0] if "_" in key else key
    # Special case handling
    if key == "GOOGLE_APPLICATION_CREDENTIALS":
        return f"{key}=PATH_TO_YOUR_GOOGLE_CREDENTIALS_FILE"
    if key == "HF_TOKEN":
        return f"{key}=YOUR_HUGGINGFACE_TOKEN_HERE"
    return f"{key}=YOUR_{service}_KEY_HERE"


def parse_requirements(project_dir):
    """Parse requirements.txt file if it exists."""
    req_file = Path(project_dir) / "requirements.txt"
    if not req_file.exists():
        print(f"Warning: No requirements.txt found in {project_dir}")
        return []

    dependencies = []
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("polars"):
                    line = line.replace("polars", "polars-lts-cpu")
                dependencies.append(line)

    return dependencies


def detect_api_keys(project_dir):
    """Attempt to detect required API keys by scanning Python files."""
    api_patterns = {
        # LLM Provider API Keys
        "HF_TOKEN": r"huggingface|hf_token",
        "OPENAI_API_KEY": r"openai|gpt",
        "ANTHROPIC_API_KEY": r"anthropic|claude",
        "MISTRAL_API_KEY": r"mistral|mistralai",
        "GEMINI_API_KEY": r"gemini|google",
        # ZenML-specific API Keys and Environment Variables
        "ZENML_STORE_API_KEY": r"zenml.*api_key|zenml_store_api_key",
        "ZENML_STORE_URL": r"zenml_store_url|zenml.*url",
        "ZENML_PROJECT_SECRET_NAME": r"zenml.*secret|secret_name",
        "ZENML_HF_USERNAME": r"zenml_hf_username|hf_username",
        "ZENML_HF_SPACE_NAME": r"zenml_hf_space_name|hf_space_name",
        # Monitoring and Logging
        "LANGFUSE_PUBLIC_KEY": r"langfuse.*public",
        "LANGFUSE_SECRET_KEY": r"langfuse.*secret",
        "LANGFUSE_HOST": r"langfuse.*host",
        # Vector Databases
        "PINECONE_API_KEY": r"pinecone",
        "SUPABASE_USER": r"supabase.*user",
        "SUPABASE_PASSWORD": r"supabase.*password",
        "SUPABASE_HOST": r"supabase.*host",
        "SUPABASE_PORT": r"supabase.*port",
        # Cloud Provider Keys
        "AWS_ACCESS_KEY_ID": r"aws.*access|aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": r"aws.*secret|aws_secret_access_key",
        "AWS_SESSION_TOKEN": r"aws.*session|aws_session_token",
        "AWS_REGION": r"aws.*region|aws_region",
        "GOOGLE_APPLICATION_CREDENTIALS": r"google.*credentials",
        # Other Service-Specific Keys
        "FIFTYONE_LABELSTUDIO_API_KEY": r"fiftyone|labelstudio",
        "NEPTUNE_API_TOKEN": r"neptune",
        "GH_ACCESS_TOKEN": r"gh_access_token|github",
    }

    detected_keys = []

    for py_file in Path(project_dir).glob("**/*.py"):
        with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().lower()
            for key, pattern in api_patterns.items():
                if re.search(pattern, content):
                    detected_keys.append(key)

    # Remove duplicates
    detected_keys = list(set(detected_keys))

    if not detected_keys:
        detected_keys = ["API_KEY=YOUR_API_KEY_HERE"]

    return [format_env_key(key) for key in detected_keys]


def detect_env_variables(project_dir, dependencies):
    """Detect which environment variables are needed based on dependencies and content."""
    env_vars = []

    # Only add POLARS_SKIP_CPU_CHECK if any polars package is in dependencies
    if any("polars" in dep.lower() for dep in dependencies):
        env_vars.append("POLARS_SKIP_CPU_CHECK=1")

    # Only add TOKENIZERS_PARALLELISM if transformers or tokenizers is used
    if any(
        dep.lower().startswith(("transform", "token")) for dep in dependencies
    ):
        env_vars.append("TOKENIZERS_PARALLELISM=false")

    # These are development convenience variables - could be made optional
    # env_vars.append("PYTHONUNBUFFERED=1")
    # env_vars.append("PYTHONDONTWRITEBYTECODE=1")

    return env_vars


def generate_dockerfile(project_path, output_dir=None):
    """Generate a Dockerfile.sandbox for the specified project."""
    if output_dir is None:
        output_dir = project_path

    base_project_name = os.path.basename(project_path)

    project_dir = Path(output_dir)
    if not project_dir.exists():
        print(f"Error: Project directory {project_dir} not found")
        return False

    # Get dependencies
    dependencies = parse_requirements(project_dir)
    if dependencies:
        formatted_deps = "\n".join(
            f'    "{dep}" \\' for dep in dependencies[:-1]
        )
        if formatted_deps:
            formatted_deps += f'\n    "{dependencies[-1]}"'
        else:
            formatted_deps = f'    "{dependencies[-1]}"'
    else:
        formatted_deps = ""

    # Detect API keys
    api_vars = detect_api_keys(project_dir)
    formatted_api_vars = '" && \\\n    echo "'.join(api_vars)

    env_vars = detect_env_variables(project_dir, dependencies)
    env_vars_block = ""
    if env_vars:
        env_vars_block = (
            "\n# Set environment variables for compatibility and performance"
        )
        for var in env_vars:
            env_vars_block += f"\nENV {var}"

    # Generate Dockerfile content
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        project_name=base_project_name,
        dependencies=formatted_deps,
        api_vars=formatted_api_vars,
        env_vars_block=env_vars_block,
    )

    # Write Dockerfile
    dockerfile_path = project_dir / "Dockerfile.sandbox"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    print(
        f"Generated Dockerfile.sandbox for {base_project_name} at {dockerfile_path}"
    )
    return True


def main():
    """Main function to parse arguments and generate Dockerfile.sandbox."""
    parser = argparse.ArgumentParser(
        description="Generate Dockerfile.sandbox for ZenML projects"
    )
    parser.add_argument("project", help="Project name")
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to project name)"
    )

    args = parser.parse_args()

    success = generate_dockerfile(args.project, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
