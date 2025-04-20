#!/usr/bin/env python3

"""Generate Dockerfile.sandbox for ZenML projects."""

import argparse
import re
import sys
from pathlib import Path

import tomli

# Dockerfile template
DOCKER_TEMPLATE = """# Sandbox base image
FROM zenmldocker/zenml-sandbox:latest

# Install uv from official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set uv environment variables for optimization
ENV UV_SYSTEM_PYTHON=1
ENV UV_COMPILE_BYTECODE=1

# Project metadata
LABEL project_name="{name}"
LABEL project_version="0.1.0"

{deps}

# Set workspace directory
WORKDIR /workspace

# Clone only the project directory and reorganize
RUN git clone --depth 1 https://github.com/zenml-io/zenml-projects.git /tmp/zenml-projects && \\
    cp -r /tmp/zenml-projects/{name}/* /workspace/ && \\
    rm -rf /tmp/zenml-projects

# VSCode settings
RUN mkdir -p /workspace/.vscode && \\
    printf '{{\\n  "workbench.colorTheme": "Default Dark Modern"\\n}}' > /workspace/.vscode/settings.json

{env_block}
"""

# Patterns to detect environment variables in code
ENV_PATTERN = re.compile(
    r"os\.(?:getenv|environ(?:\[|\\.get))\(['\"]([A-Za-z0-9_]+)['\"]\)"
)
DOTENV_PATTERN = re.compile(
    r"(?:load_dotenv|dotenv).*?['\"]([A-Za-z0-9_]+)['\"]"
)


def replace_polars(dep: str) -> str:
    """Replaces 'polars' with 'polars-lts-cpu', a CPU-optimized LTS version for container environments."""
    return (
        dep.replace("polars", "polars-lts-cpu")
        if dep.startswith("polars")
        else dep
    )


def parse_requirements(project_dir: Path) -> list[str]:
    """Parse requirements.txt and apply LTS replacement for Polars.

    Replaces 'polars' with 'polars-lts-cpu', a CPU-optimized LTS version for container environments.
    """
    req_file = project_dir / "requirements.txt"
    if not req_file.exists():
        return []
    deps = []
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            deps.append(replace_polars(line))
    return deps


def parse_pyproject(project_dir: Path) -> list[str]:
    """Parse pyproject.toml supporting PEP 621, Poetry, and PDM; replace Polars with its LTS CPU version.

    Supports dependencies under [project.dependencies], [tool.poetry.dependencies], and [tool.pdm.dependencies].
    """
    file = project_dir / "pyproject.toml"
    if not file.exists():
        return []
    try:
        data = tomli.loads(file.read_bytes())
        # PEP 621
        if deps := data.get("project", {}).get("dependencies"):  # type: ignore
            raw = deps
        # Poetry
        elif (
            poetry := data.get("tool", {})
            .get("poetry", {})
            .get("dependencies")
        ):  # type: ignore
            raw = [
                f"{n}=={v}" if isinstance(v, str) else n
                for n, v in poetry.items()
                if n != "python"
            ]
        # PDM
        elif pdm := data.get("tool", {}).get("pdm", {}).get("dependencies"):  # type: ignore
            raw = pdm
        else:
            return []
        return [replace_polars(d) for d in raw]
    except Exception as e:
        print(f"Warning: pyproject.toml parse error: {e}")
        return []


def get_dependencies(project_dir: Path) -> tuple[str, list[str]]:
    """Aggregate dependencies from requirements or pyproject and format the install block.

    Includes a warning if no dependencies are found.
    """
    deps = parse_requirements(project_dir) or parse_pyproject(project_dir)
    if not deps:
        print(f"Warning: no dependencies found in {project_dir}")
        return "# No dependencies found", []
    # build install commands
    lines = []
    lines.append("# Install dependencies with uv and cache optimization")
    lines.append("RUN --mount=type=cache,target=/root/.cache/uv \\")
    lines.append("    uv pip install --system \\")

    lines += [f'    "{d}" \\' for d in deps[:-1]] + [f'    "{deps[-1]}"']
    return "\n".join(lines), deps


def find_env_keys(project_dir: Path) -> set[str]:
    """Detect environment variable keys from .env and Python source files.

    Scans .env for explicit keys and searches code for os.getenv, os.environ, and dotenv references.
    Defaults to {'API_KEY'} if none found.
    """
    keys = set()
    env_file = project_dir / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                keys.add(line.split("=", 1)[0].strip())
    for py in project_dir.rglob("*.py"):
        txt = py.read_text(errors="ignore")
        keys |= set(ENV_PATTERN.findall(txt))
        keys |= set(DOTENV_PATTERN.findall(txt))
    return keys or {"API_KEY"}


def gen_env_block(
    project_dir: Path, keys: set[str], installed_deps: list[str]
) -> str:
    """Generate Dockerfile commands to set up .env with detected keys and runtime tweaks.

    Looks for any .env* files (like .env.example) and uses that for reference.
    Does not create a .env file if one doesn't exist.
    Adds Polars ENV only if polars-lts-cpu was installed.
    """
    lines = []

    # Look for any .env* files (.env, .env.example, etc.)
    env_files = list(project_dir.glob(".env*"))

    if env_files:
        # Use the first .env* file found
        env_file = env_files[0]
        env_file_name = env_file.name

        # Parse the existing keys from the file
        existing = set()
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and "=" in line:
                    existing.add(line.split("=", 1)[0].strip())
        except Exception:
            existing = set()

        # Copy the existing .env* file
        lines.append(f"# Copy {env_file_name}")
        lines.append(f"COPY {env_file_name} /workspace/.env")

        # Add missing keys only if we're copying a template
        missing = keys - existing
        for k in sorted(missing):
            lines.append(f'RUN echo "{k}=YOUR_{k}" >> /workspace/.env')

    # Add Polars ENV only if we actually installed polars-lts-cpu
    if any("polars-lts-cpu" in dep for dep in installed_deps):
        lines.append("ENV POLARS_SKIP_CPU_CHECK=1")

    return "\n".join(lines) if lines else ""


def generate_dockerfile(
    project_path: str,
    output_dir: str | None = None,
) -> bool:
    """Create Dockerfile.sandbox using the template, dependencies, and environment setup.

    Returns True on success, False otherwise.
    """
    out = Path(output_dir or project_path)
    if not out.exists():
        print(f"Error: {out} not found")
        return False
    name = Path(project_path).name
    deps_block, installed_deps = get_dependencies(out)
    keys = find_env_keys(out)
    env_block = gen_env_block(out, keys, installed_deps)
    content = DOCKER_TEMPLATE.format(
        name=name, deps=deps_block, env_block=env_block
    )
    (out / "Dockerfile.sandbox").write_text(content)
    print(f"Generated Dockerfile.sandbox at {out / 'Dockerfile.sandbox'}")
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        "Generate Dockerfile.sandbox for ZenML projects"
    )
    parser.add_argument("project", help="Path to the project directory")
    parser.add_argument(
        "--output-dir", help="Output directory (defaults to project path)"
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        default=True,
        help="Use uv for dependency installation (default: True)",
    )
    args = parser.parse_args()
    sys.exit(0 if generate_dockerfile(args.project, args.output_dir) else 1)


if __name__ == "__main__":
    main()
