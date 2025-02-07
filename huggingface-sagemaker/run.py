# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
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
from datetime import datetime as dt
from typing import Optional

import click
from pipelines import (
    sentinment_analysis_deploy_pipeline,
    sentinment_analysis_feature_engineering_pipeline,
    sentinment_analysis_promote_pipeline,
    sentinment_analysis_training_pipeline,
)
from zenml import Model
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

logger = get_logger(__name__)


@click.command(
    help="""
ZenML NLP project CLI v0.0.1.

Run the ZenML NLP project model training pipeline with various
options.

Examples:


  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run the pipeline without NA drop and normalization, 
  # but dropping columns [A,B,C] and keeping 10% of dataset 
  # as test set.
  python run.py --num-epochs 3 --train-batch-size 8 --eval-batch-size 8

  \b
  # Run the pipeline with Quality Gate for accuracy set at 90% for train set 
  # and 85% for test set. If any of accuracies will be lower - pipeline will fail.
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates


"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--num-epochs",
    default=1,
    type=click.INT,
    help="Number of epochs to train the model for.",
)
@click.option(
    "--train-batch-size",
    default=8,
    type=click.INT,
    help="Batch size for training the model.",
)
@click.option(
    "--eval-batch-size",
    default=8,
    type=click.INT,
    help="Batch size for evaluating the model.",
)
@click.option(
    "--learning-rate",
    default=2e-5,
    type=click.FLOAT,
    help="Learning rate for training the model.",
)
@click.option(
    "--weight-decay",
    default=0.01,
    type=click.FLOAT,
    help="Weight decay for training the model.",
)
@click.option(
    "--max-seq-length",
    default=512,
    type=click.INT,
    help="The maximum total input sequence length after tokenization.",
)
@click.option(
    "--dataset-name",
    default="tokenized_dataset",
    type=click.STRING,
    help="The name of the dataset produced by feature engineering.",
)
@click.option(
    "--dataset-version-name",
    default=None,
    type=click.STRING,
    help="Version of the dataset produced by feature engineering. "
    "If not specified, the a new version will be used.",
)
@click.option(
    "--feature-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--training-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--dataset-artifact-id",
    default=None,
    type=click.STRING,
    help="Dataset artifact id to use for training. If not specified, "
    "the latest version will be used.",
)
@click.option(
    "--tokenizer-artifact-id",
    default=None,
    type=click.STRING,
    help="Tokenizer artifact id to use for training. If not specified, "
    "the latest version will be used.",
)
@click.option(
    "--promoting-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that promotes the model to staging.",
)
@click.option(
    "--deploying-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that deploys the model to selected deployment platform.",
)
@click.option(
    "--zenml-model-name",
    default="distil_bert_sentiment_analysis",
    type=click.STRING,
    help="Name of the ZenML Model.",
)
def main(
    no_cache: bool = True,
    num_epochs: int = 3,
    train_batch_size: int = 8,
    eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_seq_length: int = 512,
    dataset_artifact_id: Optional[str] = None,
    tokenizer_artifact_id: Optional[str] = None,
    dataset_name: str = "tokenized_dataset",
    dataset_version_name: Optional[str] = None,
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    promoting_pipeline: bool = False,
    deploying_pipeline: bool = False,
    zenml_model_name: str = "distil_bert_sentiment_analysis",
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments)
      * launching the pipeline
    """

    # Run a pipeline with the required parameters. This executes
    # all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in your active ZenML stack.
    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    zenml_model = Model(
        name=zenml_model_name,
        license="Apache 2.0",
        description="Show case Model Control Plane.",
        tags=["sentiment_analysis", "huggingface"],
    )

    pipeline_args = {}

    if no_cache:
        pipeline_args["enable_cache"] = False

    # Execute Feature Engineering Pipeline
    if feature_pipeline:
        pipeline_args["model"] = zenml_model
        pipeline_args["config_path"] = os.path.join(
            config_folder, "feature_engineering_config.yaml"
        )
        run_args_feature = {
            "max_seq_length": max_seq_length,
        }
        pipeline_args["run_name"] = (
            f"sentinment_analysis_feature_engineering_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        sentinment_analysis_feature_engineering_pipeline.with_options(
            **pipeline_args
        )(**run_args_feature)
        logger.info("Feature Engineering pipeline finished successfully!")

    # Execute Training Pipeline
    if training_pipeline:
        pipeline_args["config_path"] = os.path.join(
            config_folder, "trainer_config.yaml"
        )

        run_args_train = {
            "num_epochs": num_epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_seq_length": max_seq_length,
            "dataset_artifact_id": dataset_artifact_id,
            "tokenizer_artifact_id": tokenizer_artifact_id,
        }

        # If dataset_version_name is specified, use versioned artifacts
        if dataset_version_name:
            client = Client()
            tokenized_dataset_artifact = client.get_artifact(
                dataset_name, dataset_version_name
            )
            # base tokenizer is always the same version
            # as the dataset version
            tokenized_tokenizer_artifact = client.get_artifact(
                "base_tokenizer", dataset_version_name
            )
            # Use versioned artifacts
            run_args_train["dataset_artifact_id"] = (
                tokenized_dataset_artifact.id
            )
            run_args_train["tokenizer_artifact_id"] = (
                tokenized_tokenizer_artifact.id
            )

        pipeline_args["model"] = zenml_model

        pipeline_args["run_name"] = (
            f"sentinment_analysis_training_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )

        sentinment_analysis_training_pipeline.with_options(**pipeline_args)(
            **run_args_train
        )
        logger.info("Training pipeline finished successfully!")

    # Execute Promoting Pipeline
    if promoting_pipeline:
        run_args_promoting = {}
        # Promoting pipeline always check latest version
        zenml_model = Model(
            name=zenml_model_name,
            version=ModelStages.LATEST,
        )
        pipeline_args["config_path"] = os.path.join(
            config_folder, "promoting_config.yaml"
        )

        pipeline_args["model"] = zenml_model

        pipeline_args["run_name"] = (
            f"sentinment_analysis_promoting_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        sentinment_analysis_promote_pipeline.with_options(**pipeline_args)(
            **run_args_promoting
        )
        logger.info("Promoting pipeline finished successfully!")

    if deploying_pipeline:
        pipeline_args["config_path"] = os.path.join(
            config_folder, "deploying_config.yaml"
        )

        # Deploying pipeline has new ZenML model config
        zenml_model = Model(
            name=zenml_model_name,
            version=ModelStages.PRODUCTION,
        )
        pipeline_args["model"] = zenml_model
        pipeline_args["enable_cache"] = False
        run_args_deploying = {}
        pipeline_args["run_name"] = (
            f"sentinment_analysis_deploy_pipeline_run_{dt.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        )
        sentinment_analysis_deploy_pipeline.with_options(**pipeline_args)(
            **run_args_deploying
        )
        logger.info("Deploying pipeline finished successfully!")


if __name__ == "__main__":
    main()
