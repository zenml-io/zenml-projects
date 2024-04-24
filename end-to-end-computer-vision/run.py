# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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
from uuid import UUID

import click
from zenml import Model
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

from pipelines.data_export import export_for_training
from pipelines.data_ingestion import data_ingestion
from pipelines.fifty_one import export_predictions
from pipelines.inference import inference
from pipelines.training import training
from utils.constants import ZENML_MODEL_NAME

logger = get_logger(__name__)

REMOTE_STACK_ID = UUID("20ed5311-ffc6-45d0-b339-6ec35af9501e")


@click.command()
@click.option(
    "--ingest",
    "-ig",
    "ingest_data_pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the data ingestion pipeline, that takes the dataset"
    "from huggingface and uploads it into label studio.",
)
@click.option(
    "--export",
    "-e",
    "export_pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that exports the dataset from "
    "labelstudio.",
)
@click.option(
    "--training",
    "-t",
    "training_pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference",
    "-i",
    "inference_pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)
@click.option(
    "--fiftyone",
    "-fo",
    "fiftyone",
    is_flag=True,
    default=False,
    help="Whether to launch the FiftyOne app pipeline.",
)
@click.option(
    "--stack",
    "-s",
    "stack",
    required=False,
    type=click.Choice(["alexej", "hamza", "alex"]),
    help="The stack to use for the pipeline.",
)
@click.option(
    "--local",
    "-l",
    "train_local",
    is_flag=True,
    default=False,
    help="Whether to train local.",
)
def main(
    ingest_data_pipeline: bool = False,
    export_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    fiftyone: bool = False,
    stack: UUID = "alexej",
    train_local: bool = False,
):
    # TODO: remove all this :)
    if stack == "hamza":
        stack_id = UUID("cca5eaf7-0309-413d-89ff-1cd371b7d27c")
    elif stack == "alex":
        stack_id = UUID("fcf840ac-addd-4de3-a3e4-a1015f7bb96c")
    else:
        stack_id = UUID("7cda3cec-6744-48dc-8bdc-f102242a26d2")

    client = Client()

    if ingest_data_pipeline:
        data_ingestion.with_options(config_path="configs/ingest_data.yaml")()

    if export_pipeline:
        client.activate_stack(stack_id)

        # Export data from label studio
        export_for_training.with_options(
            config_path="configs/data_export_alexej.yaml"
        )()

        # Promote Model to staging
        latest_model = Model(name=ZENML_MODEL_NAME, version=ModelStages.LATEST)
        latest_model.set_stage(stage=ModelStages.STAGING, force=True)

    if training_pipeline and train_local:
        client.activate_stack(stack_id)

        # Train model on data
        training.with_options(config_path="configs/training.yaml")()

    if training_pipeline and not train_local:
        client.activate_stack(REMOTE_STACK_ID)

        # Train model on data
        training.with_options(config_path="configs/training_gpu.yaml")()

    if inference_pipeline:
        client.activate_stack(REMOTE_STACK_ID)

        inference.with_options(config_path="configs/cloud_inference.yaml")()

    if fiftyone:
        client.activate_stack(stack_id)

        export_predictions.with_options(config_path="configs/fiftyone.yaml")()


if __name__ == "__main__":
    main()
