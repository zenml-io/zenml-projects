# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from pathlib import Path

import click
from zenml.logger import get_logger

from pipelines import (
    deployment,
    feature_engineering,
    training,
)
from steps.post_run_annex import generate_annex_iv_documentation

logger = get_logger(__name__)


@click.command(
    help="""
EU AI Act Compliance Credit Scoring Project.

Run various pipelines with EU AI Act compliance features.

Examples:

  \b
  # Run the feature engineering pipeline
    python run.py --feature
  
  \b
  # Run the training pipeline with fairness checks
    python run.py --train

  \b
  # Run the deployment pipeline with human approval
    python run.py --deploy

  \b
  # Run complete end-to-end workflow with documentation
    python run.py --all --generate-docs
"""
)
@click.option(
    "--feature",
    is_flag=True,
    default=False,
    help="Run the feature engineering pipeline (Articles 10, 12).",
)
@click.option(
    "--train",
    is_flag=True,
    default=False,
    help="Run the model training pipeline (Articles 9, 11, 15).",
)
@click.option(
    "--deploy",
    is_flag=True,
    default=False,
    help="Run the deployment pipeline (Articles 14, 17, 18).",
)
@click.option(
    "--all",
    is_flag=True,
    default=False,
    help="Run complete workflow (feature → training → deployment).",
)
@click.option(
    "--config-dir",
    default="configs",
    type=click.STRING,
    help="Directory containing configuration files.",
)
@click.option(
    "--model-id",
    default=None,
    type=click.STRING,
    help="ID of model to use for deployment (required for deployment).",
)
@click.option(
    "--generate-docs",
    is_flag=True,
    default=True,
    help="Generate EU AI Act Annex IV documentation.",
)
@click.option(
    "--auto-approve",
    is_flag=True,
    default=False,
    help="Auto-approve deployment (for CI/CD pipelines).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for pipeline runs.",
)
def main(
    feature: bool = False,
    train: bool = False,
    deploy: bool = False,
    all: bool = False,
    config_dir: str = "configs",
    model_id: str = None,
    generate_docs: bool = False,
    auto_approve: bool = False,
    no_cache: bool = False,
):
    """Main entry point for EU AI Act compliance pipelines.

    Relies on configuration files in the specified directory
    for pipeline parameters and settings.
    """
    # Get full path to config directory
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = Path(os.path.dirname(os.path.realpath(__file__))) / config_dir

    # Ensure config directory exists
    if not config_dir.exists():
        raise ValueError(f"Configuration directory {config_dir} not found")

    # Handle auto-approve setting for deployment
    if auto_approve:
        os.environ["DEPLOY_APPROVAL"] = "y"
        os.environ["APPROVER"] = "automated_ci"
        os.environ["APPROVAL_RATIONALE"] = "Automatic approval via --auto-approve flag"

    # Common pipeline options
    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False

    # Track outputs for chaining pipelines
    outputs = {}

    # Ignore WhyLogs optional usage-telemetry API
    os.environ["WHYLOGS_NO_ANALYTICS"] = "True"

    # Run complete workflow if requested
    if all:
        logger.info("Starting complete EU AI Act compliant workflow...")

        # Run feature engineering pipeline
        config_path = config_dir / "feature_engineering.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        fe_pipeline = feature_engineering.with_options(**pipeline_args)
        train_df, test_df, preprocess_pipeline, *_ = fe_pipeline()
        logger.info("✅ Feature engineering pipeline completed")

        # Run training pipeline
        config_path = config_dir / "training.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        training_pipeline = training.with_options(**pipeline_args)
        training_results = training_pipeline(
            train_df=train_df,
            test_df=test_df,
        )
        logger.info("✅ Training pipeline completed")

        # Run deployment pipeline
        config_path = config_dir / "deployment.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        deployment_pipeline = deployment.with_options(**pipeline_args)
        deployment_pipeline(
            model_path=training_results["model_path"],
            evaluation_results=training_results["evaluation"],
            risk_info=training_results["risk"],
            preprocess_pipeline=preprocess_pipeline,
        )
        logger.info("✅ Deployment pipeline completed")

        logger.info("🎉 Complete EU AI Act compliant workflow finished successfully!")
        return

    # Run feature engineering pipeline if requested
    if feature:
        config_path = config_dir / "feature_engineering.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        fe_pipeline = feature_engineering.with_options(**pipeline_args)
        train_df, test_df, preprocess_pipeline, *_ = fe_pipeline()

        # Store for potential chaining
        outputs["train_df"] = train_df
        outputs["test_df"] = test_df
        outputs["preprocess_pipeline"] = preprocess_pipeline

        logger.info("✅ Feature engineering pipeline completed")

    # Run training pipeline if requested
    if train:
        config_path = config_dir / "training.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        train_args = {}

        # Use outputs from previous pipeline if available
        if "train_df" in outputs and "test_df" in outputs:
            train_args["train_df"] = outputs["train_df"]
            train_args["test_df"] = outputs["test_df"]

        training_pipeline = training.with_options(**pipeline_args)
        training_results = training_pipeline(**train_args)

        # Store for potential chaining
        outputs["model_path"] = training_results["model_path"]
        outputs["evaluation"] = training_results["evaluation"]
        outputs["risk"] = training_results["risk"]

        logger.info("✅ Training pipeline completed")

    # Run deployment pipeline if requested
    if deploy:
        config_path = config_dir / "deployment.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        deploy_args = {}

        # Use model from training pipeline or specified model_id
        if "model_path" in outputs:
            deploy_args["model_path"] = outputs["model_path"]
        elif model_id:
            # Load model by ID
            deploy_args["model_path"] = model_id
        else:
            raise ValueError(
                "Model ID must be provided for deployment when not chaining pipelines. "
                "Use --model-id to specify."
            )

        # Add evaluation and risk info if available
        if "evaluation" in outputs:
            deploy_args["evaluation_results"] = outputs["evaluation"]

        if "risk" in outputs:
            deploy_args["risk_info"] = outputs["risk"]

        if "preprocess_pipeline" in outputs:
            deploy_args["preprocess_pipeline"] = outputs["preprocess_pipeline"]

        deployment_pipeline = deployment.with_options(**pipeline_args)
        deployment_pipeline(**deploy_args)

        logger.info("✅ Deployment pipeline completed")

    # If no pipeline specified, show help
    if not any([feature, train, deploy, all]):
        ctx = click.get_current_context()
        click.echo(ctx.get_help())


if __name__ == "__main__":
    main()
