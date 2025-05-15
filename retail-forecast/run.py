import click
import logging
from pipelines.inference_pipeline import inference_pipeline
from pipelines.training_pipeline import training_pipeline
from zenml import Model
from logging_config import configure_logging

logger = logging.getLogger(__name__)


@click.command(
    help="""
RetailForecast - Simple Retail Demand Forecasting with ZenML and Prophet

Run a simplified retail demand forecasting pipeline using Facebook Prophet.

Examples:

  \b
  # Run the training pipeline with default parameters
  python run.py

  \b
  # Run with custom parameters
  python run.py --forecast-periods 60 --test-size 0.3
  
  \b
  # Run the inference pipeline
  python run.py --inference
"""
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
    no_cache: bool = False,
    inference: bool = False,
    log_file: str = None,
    debug: bool = False,
):
    """Run a simplified retail forecasting pipeline with ZenML.

    Args:
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

    logger.info("\n" + "=" * 80)
    # Run the appropriate pipeline
    if inference:
        logger.info("Running retail forecasting inference pipeline...")

        # Create a new version of the model
        model = Model(
            name="retail_forecast_model",
            description="A retail forecast model trained on the sales data",
            version="production",
        )
        inference_pipeline.with_options(model=model, **pipeline_options)()
    else:
        # Create a new version of the model
        model = Model(
            name="retail_forecast_model",
            description="A retail forecast model trained on the sales data",
        )

        logger.info("Running retail forecasting training pipeline...")
        training_pipeline.with_options(model=model, **pipeline_options)()
    logger.info("=" * 80 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
