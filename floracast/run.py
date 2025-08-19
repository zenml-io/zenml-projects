"""
Entry point for running FloraCast pipelines using ZenML e2e pattern.
"""

import click
from datetime import datetime
from pipelines import batch_inference_pipeline, train_forecast_pipeline
from zenml.logger import get_logger
from materializers import DartsTimeSeriesMaterializer, TFTModelMaterializer

logger = get_logger(__name__)


@click.command()
@click.option(
    "--config",
    "-c", 
    type=click.Path(exists=True, dir_okay=False),
    default="configs/local.yaml",
    help="Path to configuration YAML file"
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Choice(["train", "inference", "both"]),
    default="train",
    help="Pipeline to run"
)
def main(config: str, pipeline: str):
    """Run FloraCast forecasting pipelines using ZenML with_options pattern."""
    
    # Generate run name with timestamp
    run_name = f"floracast_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    
    logger.info(f"Starting FloraCast run: {run_name}")
    logger.info(f"Using configuration: {config}")
    
    try:
        if pipeline in ["train", "both"]:
            logger.info("Starting training pipeline...")
            train_forecast_pipeline.with_options(
                config_path=config,
                run_name=f"{run_name}_train"
            )()
            logger.info("Training pipeline completed successfully!")
        
        if pipeline in ["inference", "both"]:
            logger.info("Starting batch inference pipeline...")
            batch_inference_pipeline.with_options(
                config_path=config,
                run_name=f"{run_name}_inference"  
            )()
            logger.info("Batch inference pipeline completed successfully!")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()