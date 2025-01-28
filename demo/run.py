import os
from typing import Optional

import click
import yaml
from pipelines.cifar10_pipeline import cifar10_pipeline

from zenml.client import Client
from zenml.config.schedule import Schedule
from zenml.integrations.neptune.experiment_trackers import (
    NeptuneExperimentTracker,
)


@click.command(
    help="""
ZenML CIFAR10 Training Demo CLI.

Run the ZenML CIFAR10 image classification training pipeline.

Examples:

  \b
  # Run the pipeline with default config
    python run.py
  
  \b
  # Run the pipeline with custom config
    python run.py --config custom_config.yaml

  \b
  # Run without caching
    python run.py --no-cache
"""
)
@click.option(
    "--config-path",
    type=str,
    default="configs/config.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
def main(config_path: Optional[str] = None, no_cache: bool = False) -> None:
    """Main entry point for the pipeline execution.

    Args:
        config: Path to the YAML config file.
        no_cache: If True, disable caching.
    """
    if not config_path:
        raise RuntimeError("Config file is required to run the pipeline.")

    # Ensure config path is absolute
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            config_path
        )

    # Load configuration
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Ensure neptune experiment tracker is active
    stack = Client().active_stack
    if not isinstance(stack.experiment_tracker, NeptuneExperimentTracker):
        raise RuntimeError(
            "This pipeline requires an Neptune experiment tracker in the active stack. "
            "Please run: zenml experiment-tracker register neptune"
        )
    
    # Run the pipeline
    pipeline_args = {"enable_cache": not no_cache}
    pipeline_args["config_path"] = config_path
    metrics = cifar10_pipeline.with_options(**pipeline_args,)(
        batch_size=config_dict["parameters"]["batch_size"],
        val_split=config_dict["parameters"]["val_split"],
        dataset_fraction=config_dict["parameters"]["dataset_fraction"],
        epochs=config_dict["parameters"]["epochs"],
        learning_rate=config_dict["parameters"]["learning_rate"],
    )
    
    click.echo("Training completed!")
    click.echo(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main() 