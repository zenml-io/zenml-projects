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

from typing import Optional

import click
from utils import (
    load_config,
    with_setup_environment,
)


@click.group(
    help="""
ZenML Article Classification CLI.

Run the ZenML ModernBERT finetuning and article classification pipelines.

Examples:

  # Run Classification pipeline
    python run.py classify

  # Run training pipeline with local config
    python run.py train --config base_config.yaml

  # Run training pipeline with remote config
    python run.py train --config remote_finetune.yaml

  # Run model deployment
    python run.py deploy

  # Run model comparison
    python run.py compare
"""
)
def cli():
    pass


@cli.command()
@click.option(
    "--config",
    type=str,
    default="base_config.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--mode",
    type=click.Choice(["evaluation", "augmentation"]),
    default="evaluation",
    help="Mode for classification pipeline: evaluation or augmentation.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@with_setup_environment
def classify(config: str, mode: str, no_cache: bool):
    """Run the classification pipeline."""
    from pipelines import classification_pipeline

    config_data = load_config(config)
    config_data["steps"]["classify"]["classification_type"] = mode

    pipeline_args = {"enable_cache": not no_cache}
    classification_pipeline.with_options(**pipeline_args)(config=config_data)


@cli.command()
@click.option(
    "--config",
    type=str,
    default="base_config.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--save-test",
    is_flag=True,
    default=False,
    help="Save test set to disk for later evaluation.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@with_setup_environment
def train(config: str, save_test: bool, no_cache: bool):
    """Run the training pipeline."""
    from pipelines import training_pipeline

    config_data = load_config(config)

    pipeline_args = {"enable_cache": not no_cache}
    training_pipeline.with_options(**pipeline_args)(
        config=config_data, save_to_disk=save_test
    )


@cli.command()
@click.option(
    "--config",
    type=str,
    default="base_config.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--model-dir",
    type=str,
    help="Path to the model directory.",
)
@click.option(
    "--tokenizer-dir",
    type=str,
    help="Path to the tokenizer directory.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@with_setup_environment
def deploy(
    config: str,
    model_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    no_cache: bool = False,
):
    """Run the deployment pipeline."""
    from pipelines import deployment_pipeline

    config_data = load_config(config)
    pipeline_args = {"enable_cache": not no_cache}

    call_args = {"config": config_data}
    if model_dir:
        call_args["model_dir"] = model_dir
    if tokenizer_dir:
        call_args["tokenizer_dir"] = tokenizer_dir

    deployment_pipeline.with_options(**pipeline_args)(**call_args)


@cli.command()
@click.option(
    "--config",
    type=str,
    default="base_config.yaml",
    help="Path to the YAML config file.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@with_setup_environment
def compare(config: str, no_cache: bool):
    """Run the model comparison pipeline."""
    from pipelines import model_comparison_pipeline

    config_data = load_config(config)
    pipeline_args = {"enable_cache": not no_cache}
    model_comparison_pipeline.with_options(**pipeline_args)(config=config_data)


if __name__ == "__main__":
    cli()
