"""Main module to run the Bank Subscription Prediction pipeline."""

import click
import logging
from pipelines.training_pipeline import bank_subscription_training_pipeline

logger = logging.getLogger(__name__)


@click.command(
    help="""
Bank Subscription Prediction - Predict Term Deposit Subscriptions with ZenML

Run a machine learning pipeline to predict which bank customers are likely to 
subscribe to a term deposit.

Examples:

  \b
  # Run the training pipeline with default parameters
  python run.py

  \b
  # Run with a specific configuration file
  python run.py --config configs/more_trees.yaml

  \b
  # Run with debugging enabled
  python run.py --debug
"""
)
@click.option(
    "--config",
    type=str,
    default="configs/baseline.yaml",
    help="Path to the configuration YAML file",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run",
)
def main(
    config: str = "configs/baseline.yaml",
    no_cache: bool = False,
):
    """Run the bank subscription prediction pipeline.

    Args:
        config: Path to the configuration YAML file
        no_cache: Disable caching for the pipeline run
    """
    pipeline_options = {}
    if no_cache:
        pipeline_options["enable_cache"] = False

    logger.info("\n" + "=" * 80)
    logger.info("Running bank subscription prediction pipeline...")

    # Run the pipeline with the specified config
    logger.info(f"Using configuration from: {config}")
    pipeline_options["config_path"] = config

    run = bank_subscription_training_pipeline.with_options(
        **pipeline_options
    )()

    logger.info("=" * 80 + "\n")
    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline completed successfully! Run ID: {run.id}")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
