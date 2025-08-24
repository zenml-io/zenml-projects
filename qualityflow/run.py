"""
Entry point for running QualityFlow test generation pipeline.
"""

from pathlib import Path

import click
from pipelines import generate_and_evaluate
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "-c", 
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    required=False,
    help="Path to configuration YAML file. Defaults to configs/experiment.default.yaml",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable pipeline caching and force fresh execution",
)
def main(config: str | None, no_cache: bool):
    """Run QualityFlow test generation and coverage analysis pipeline.
    
    Simple pipeline that generates tests using LLM, runs them, measures coverage,
    and compares results against baseline approaches.
    """

    project_root = Path(__file__).parent
    default_config = project_root / "configs" / "experiment.default.yaml"
    chosen_config = config or str(default_config)

    try:
        logger.info(f"Starting QualityFlow pipeline with config: {chosen_config}")
        pipeline_instance = generate_and_evaluate.with_options(
            config_path=chosen_config, 
            enable_cache=not no_cache
        )
        pipeline_instance()
        logger.info("QualityFlow pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()