"""CLI entrypoint for the LLM Code Evaluation pipeline.

Usage:
    python run.py                          # default config (2 models)
    python run.py --config configs/deepseek_only.yaml   # quick test
    python run.py --config configs/full_comparison.yaml  # all models
    python run.py --no-cache               # disable caching
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from materializers.register_materializers import register_materializers
from pipelines.code_eval_pipeline import code_eval_pipeline

logger = logging.getLogger(__name__)


@click.command(
    help="Run the LLM code evaluation pipeline.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML config file (default: configs/default.yaml)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable ZenML step caching for this run.",
)
def main(config_path: str | None, no_cache: bool) -> None:
    """Run the LLM code evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Register custom materializers before pipeline execution
    register_materializers()

    # Resolve config path
    project_root = Path(__file__).parent
    if config_path is None:
        config_path = str(project_root / "configs" / "default.yaml")
    elif not Path(config_path).is_absolute():
        config_path = str(project_root / config_path)

    logger.info("Running code_eval_pipeline with config: %s", config_path)
    logger.info("Caching: %s", "disabled" if no_cache else "enabled")

    pipeline_instance = code_eval_pipeline.with_options(
        config_path=config_path,
        enable_cache=not no_cache,
    )
    pipeline_instance()


if __name__ == "__main__":
    main()
