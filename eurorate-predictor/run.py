import os

import click
from pipelines import (
    ecb_predictor_etl_pipeline,
    ecb_predictor_feature_engineering_pipeline,
    ecb_predictor_model_training_pipeline,
)
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)
client = Client()


@click.command(
    help="""
ZenML ECB Interest Rate project.

Run the ZenML pipelines with basic options.

Examples:

  \b
  # Run the ETL pipeline
    python run.py --etl

  \b
  # Run the feature engineering pipeline with a specific transformed dataset version
    python run.py --feature --transformed_dataset_version "v1"
  
  \b
  # Run the model training pipeline with a specific augmented dataset version
    python run.py --training --augmented_dataset_version "v1"

"""
)
@click.option(
    "--etl",
    is_flag=True,
    default=False,
    help="Whether to run the ETL pipeline.",
)
@click.option(
    "--feature",
    is_flag=True,
    default=False,
    help="Whether to run the feature engineering pipeline.",
)
@click.option(
    "--training",
    is_flag=True,
    default=False,
    help="Whether to run the model training pipeline.",
)
@click.option(
    "--mode",
    type=click.Choice(["develop", "production"]),
    default="develop",
    help="The mode in which to run the pipelines.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--transformed_dataset_version",
    type=str,
    default=None,
    help="Version of the transformed dataset to use for feature engineering. Defaults to latest.",
)
@click.option(
    "--augmented_dataset_version",
    type=str,
    default=None,
    help="Version of the augmented dataset to use for model training. Defaults to latest.",
)
def main(
    etl: bool = False,
    feature: bool = False,
    training: bool = False,
    mode: str = "develop",
    no_cache: bool = False,
    transformed_dataset_version: str = None,
    augmented_dataset_version: str = None,
):
    """Main entry point for the pipeline execution."""
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Execute ETL Pipeline
    if etl:
        pipeline_args = {}
        run_args_etl = {"mode": mode}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"etl_{mode}.yaml"
        )
        ecb_predictor_etl_pipeline.with_options(**pipeline_args)(
            **run_args_etl
        )
        logger.info("ETL pipeline finished successfully!\n")

    # Execute Feature Engineering Pipeline
    if feature:
        run_args_feature = {"mode": mode}
        try:
            transformed_dataset_artifact_version = client.get_artifact_version(
                "ecb_transformed_dataset", transformed_dataset_version
            )
            run_args_feature["transformed_dataset_id"] = str(
                transformed_dataset_artifact_version.id
            )
        except KeyError:
            logger.error(
                f"Transformed dataset version '{transformed_dataset_version}' not found. Using the latest version."
            )
            raise

        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"feature_engineering_{mode}.yaml"
        )
        ecb_predictor_feature_engineering_pipeline.with_options(
            **pipeline_args
        )(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!\n")

    # Execute Model Training Pipeline
    if training:
        run_args_train = {"mode": mode}
        try:
            augmented_dataset_artifact_version = client.get_artifact_version(
                "ecb_augmented_dataset", augmented_dataset_version
            )
            run_args_train["augmented_dataset_id"] = str(
                augmented_dataset_artifact_version.id
            )
        except KeyError:
            logger.error(
                f"Augmented dataset version '{augmented_dataset_version}' not found. Using the latest version."
            )
            raise

        pipeline_args = {}
        if no_cache:
            pipeline_args["enable_cache"] = False
        pipeline_args["config_path"] = os.path.join(
            config_folder, f"training_{mode}.yaml"
        )
        ecb_predictor_model_training_pipeline.with_options(**pipeline_args)(
            **run_args_train
        )
        logger.info("Model Training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
