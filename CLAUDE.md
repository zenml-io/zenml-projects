# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo of production-grade ML projects built with [ZenML](https://zenml.io). Each subdirectory is a self-contained project demonstrating MLOps best practices across domains: LLMOps, Computer Vision, traditional MLOps, and Data Science.

## Essential Commands

```bash
# REQUIRED before any commit - formats entire repo
bash scripts/format.sh

# Run a specific project
cd <project-name>/
pip install -r requirements.txt  # or: uv pip install -r requirements.txt
python run.py                     # most projects use this entry point
python run.py --help              # check project-specific options

# View ZenML dashboard
zenml up
```

## Code Style

- **Formatter**: Ruff (line length: 79 chars)
- **Docstrings**: Google style
- **Imports**: Sorted automatically by ruff; group by stdlib → third-party → local

The `scripts/format.sh` script handles all formatting (unused import removal, import sorting, code formatting).

## Project Structure Pattern

Each project follows this structure:
```
<project>/
├── run.py              # Main entry point with CLI
├── requirements.txt    # Dependencies
├── configs/            # YAML pipeline configurations
├── pipelines/          # ZenML @pipeline definitions
├── steps/              # ZenML @step definitions
├── utils/              # Helper functions
└── materializers/      # Custom artifact serializers (optional)
```

## ZenML Patterns

- Pipelines use `@pipeline` decorator, steps use `@step` decorator
- Configuration is typically YAML-based in `configs/` directory
- Run with `--no-cache` flag to disable ZenML caching during development
- Artifacts are automatically versioned and tracked

## Adding New Projects

See `ADDING_PROJECTS.md` for the complete guide. Key requirements:
- Include `requirements.txt`
- Add comprehensive `README.md`
- Add project to the table in root `README.md`
- Run `bash scripts/format.sh` before committing
