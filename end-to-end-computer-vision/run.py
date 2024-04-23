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

from pipelines import data_export
from pipelines.cloud_inference import cloud_inference
from pipelines.local_inference import fifty_one
from pipelines.training import training

logger = get_logger(__name__)


@click.command()
@click.option(
    "--feature",
    "-f",
    "feature_pipeline",
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
def main(
    feature_pipeline: bool = False,
    training_pipeline: bool = False,
    inference_pipeline: bool = False,
    fiftyone: bool = False,
):
    client = Client()

    if feature_pipeline:
        client.activate_stack(UUID("7cda3cec-6744-48dc-8bdc-f102242a26d2"))

        # Export data from label studio
        data_export.with_options(
            config_path="configs/data_export_alexej.yaml"
        )()

        # Promote Model to staging
        latest_model = Model(
            name="Yolo_Object_Detection", version=ModelStages.LATEST
        )
        latest_model.set_stage(stage=ModelStages.STAGING, force=True)

    if training_pipeline:
        client.activate_stack(UUID("20ed5311-ffc6-45d0-b339-6ec35af9501e"))

        # Train model on data
        training.with_options(config_path="configs/training_gpu.yaml")()
        # training.with_options(config_path="configs/training.yaml")()

    if inference_pipeline:
        client.activate_stack(UUID("7cda3cec-6744-48dc-8bdc-f102242a26d2"))

        cloud_inference.with_options(config_path="configs/inference.yaml")()

    if fiftyone:
        # client.activate_stack(UUID("7cda3cec-6744-48dc-8bdc-f102242a26d2"))

        fifty_one.with_options(config_path="configs/fiftyone.yaml")()


if __name__ == "__main__":
    main()
