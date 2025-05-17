import logging

import click
from logging_config import configure_logging
from pipelines.inference_pipeline import inference_pipeline
from pipelines.training_pipeline import training_pipeline

logger = logging.getLogger(__name__)


@click.command(
    help="""
RetailForecast - Simple Retail Demand Forecasting with ZenML and Prophet

Run a simplified retail demand forecasting pipeline using Facebook Prophet.

Examples:

  \b
  # Run the training pipeline with default training config
  python run.py

  \b
  # Run with a specific training configuration file
  python run.py --config configs/training.yaml
  
  \b
  # Run the inference pipeline with default inference config
  python run.py --inference
"""
)
@click.option(
    "--config",
    type=str,
    default=None,
    help="Path to the configuration YAML file",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
@click.option(
    "--inference",
    is_flag=True,
    default=False,
    help="Run the inference pipeline instead of the training pipeline",
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Path to log file (if not provided, logs only go to console)",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug logging",
)
def main(
    config: str = None,
    no_cache: bool = False,
    inference: bool = False,
    log_file: str = None,
    debug: bool = False,
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
        config: Path to the configuration YAML file
        no_cache: Disable caching for the pipeline run
        inference: Run the inference pipeline instead of the training pipeline
        log_file: Path to log file
        debug: Enable debug logging
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    configure_logging(level=log_level, log_file=log_file)

    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    # Select default config based on pipeline type if not specified
    if config is None:
        config = (
            "configs/inference.yaml" if inference else "configs/training.yaml"
        )

    # Set config path
    pipeline_options["config_path"] = config

    logger.info("\n" + "=" * 80)
    logger.info(f"Using configuration from: {config}")

    # Run the appropriate pipeline
    if inference:
        logger.info("Running retail forecasting inference pipeline...")
        run = inference_pipeline.with_options(**pipeline_options)()
    else:
        logger.info("Running retail forecasting training pipeline...")
        run = training_pipeline.with_options(**pipeline_options)()

    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
