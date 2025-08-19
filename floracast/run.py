"""
Entry point for running FloraCast pipelines using ZenML e2e pattern.
"""

import click
from pathlib import Path
from pipelines import batch_inference_pipeline, train_forecast_pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    required=False,
    help=(
        "Path to configuration YAML file. If not provided, selects a sensible default "
        "based on the chosen pipeline (training.yaml for train, inference.yaml for inference)."
    ),
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Choice(["train", "inference", "both"]),
    default="train",
    help="Pipeline to run",
)
def main(config: str | None, pipeline: str):
    """Run FloraCast forecasting pipelines using ZenML with_options pattern.

    When --config is omitted, a default is chosen per pipeline:
    - train: floracast/configs/training.yaml
    - inference: floracast/configs/inference.yaml
    - both: training uses training.yaml, inference uses inference.yaml
    """

    project_root = Path(__file__).parent
    default_training_config = project_root / "configs" / "training.yaml"
    default_inference_config = project_root / "configs" / "inference.yaml"

    try:
        if pipeline in ["train", "both"]:
            chosen_config = config or str(default_training_config)
            logger.info(
                f"Starting training pipeline with config: {chosen_config}"
            )
            train_forecast_pipeline.with_options(config_path=chosen_config)()
            logger.info("Training pipeline completed successfully!")

        if pipeline in ["inference", "both"]:
            chosen_config = config or str(default_inference_config)
            logger.info(
                f"Starting batch inference pipeline with config: {chosen_config}"
            )
            batch_inference_pipeline.with_options(config_path=chosen_config)()
            logger.info("Batch inference pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
