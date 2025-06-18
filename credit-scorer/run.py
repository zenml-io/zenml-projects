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
from src.pipelines import (
    deployment,
    feature_engineering,
    training,
)
from zenml.logger import get_logger

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
    default="src/configs",
    type=click.STRING,
    help="Directory containing configuration files.",
)
@click.option(
    "--manual-approve",
    is_flag=True,
    default=False,
    help="Require manual approval for deployment (disables auto-approve).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for pipeline runs.",
)
@click.option(
    "--enable-slack",
    is_flag=True,
    default=False,
    help="Enable Slack notifications in deployment (requires Modal secrets setup).",
)
def main(
    feature: bool = False,
    train: bool = False,
    deploy: bool = False,
    all: bool = False,
    config_dir: str = "src/configs",
    manual_approve: bool = False,
    no_cache: bool = False,
    enable_slack: bool = False,
):
    """Main entry point for EU AI Act compliance pipelines.

    Relies on configuration files in the specified directory
    for pipeline parameters and settings.
    """
    # Get full path to config directory
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = (
            Path(os.path.dirname(os.path.realpath(__file__))) / config_dir
        )
    if not config_dir.exists():
        raise ValueError(f"Configuration directory {config_dir} not found")

    # Handle approval setting for deployment (auto-approve is now default)
    auto_approve = not manual_approve
    if auto_approve:
        os.environ["DEPLOY_APPROVAL"] = "y"
        os.environ["APPROVER"] = "automated_ci"
        os.environ["APPROVAL_RATIONALE"] = (
            "Automatic approval (default behavior)"
        )

    # Handle Slack setting for deployment (Slack disabled by default)
    if enable_slack:
        os.environ["ENABLE_SLACK"] = "true"

    # Common pipeline options
    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False

    # Handle --all flag first
    if all:
        feature = train = deploy = True

    # Track outputs for chaining pipelines
    outputs = {}

    # Ignore WhyLogs optional usage-telemetry API
    os.environ["WHYLOGS_NO_ANALYTICS"] = "True"

    # Run feature engineering pipeline
    if feature:
        config_path = config_dir / "feature_engineering.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        run_args = {}
        fe_pipeline = feature_engineering.with_options(**pipeline_args)
        train_df, test_df, sk_pipeline, whylogs_profile, *_ = fe_pipeline(
            **run_args
        )

        logger.info("✅ Feature engineering pipeline finished successfully!")

        # Store for potential chaining
        outputs["train_df"] = train_df
        outputs["test_df"] = test_df
        outputs["sk_pipeline"] = sk_pipeline
        outputs["whylogs_profile"] = whylogs_profile

    # Run training pipeline
    if train:
        config_path = config_dir / "training.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        train_args = {}

        # Don't pass DataFrame artifacts directly - let training pipeline fetch them
        # from artifact store via Client.get_artifact_version() as designed

        training_pipeline = training.with_options(**pipeline_args)
        (
            model,
            eval_results,
            eval_visualization,
            risk_scores,
            risk_visualization,
            *_,
        ) = training_pipeline(**train_args)

        # Store for potential chaining
        outputs["model"] = model
        outputs["evaluation_results"] = eval_results
        outputs["eval_visualization"] = eval_visualization
        outputs["risk_scores"] = risk_scores

        logger.info("✅ Training pipeline completed")

    # Run deployment pipeline
    if deploy:
        config_path = config_dir / "deployment.yaml"
        if config_path.exists():
            pipeline_args["config_path"] = str(config_path)

        deploy_args = {}

        # Don't pass artifacts directly - let deployment pipeline fetch them
        # from artifact store via Client.get_artifact_version() as designed

        deployment.with_options(**pipeline_args)(**deploy_args)

        logger.info("✅ Deployment pipeline completed")

    # If no pipeline specified, show help
    if not any([feature, train, deploy]):
        ctx = click.get_current_context()
        click.echo(ctx.get_help())


if __name__ == "__main__":
    main()
