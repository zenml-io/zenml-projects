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
from zenml.client import Client
from zenml.enums import ModelStages
from zenml.logger import get_logger

from pipelines.data_export import data_export_pipeline
from pipelines.data_ingestion import data_ingestion_pipeline
from pipelines.inference import inference_pipeline
from pipelines.training import training_pipeline
from utils.constants import PREDICTIONS_DATASET_ARTIFACT_NAME, ZENML_MODEL_NAME

logger = get_logger(__name__)


@click.command()
@click.option(
    "--ingest",
    "-ig",
    "ingest_data",
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
    "train",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--inference",
    "-i",
    "batch_inference",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that performs inference.",
)
@click.option(
    "--fiftyone",
    "-f",
    "fiftyone",
    is_flag=True,
    default=False,
    help="Whether to launch the FiftyOne app.",
)
@click.option(
    "--local",
    "-l",
    "train_local",
    is_flag=True,
    default=False,
    help="Whether to train local or an a remote orchestrator/ step operator.",
)
def main(
    ingest_data: bool = False,
    export_pipeline: bool = False,
    train: bool = False,
    batch_inference: bool = False,
    fiftyone: bool = False,
    train_local: bool = False,
):
    client = Client()

    if ingest_data:
        if not client.active_stack.orchestrator.config.is_local:
            raise RuntimeError(
                "The implementation of this pipeline "
                "requires that you are running on a local "
                "machine with data being persisted in the local "
                "filesystem across multiple steps. Please "
                "switch to a stack that contains a local "
                "orchestrator and a local label-studio "
                "annotator. See the README for more information "
                "on this setup."
            )

        data_ingestion_pipeline.with_options(
            config_path="configs/ingest_data.yaml"
        )()

    if export_pipeline:
        if not client.active_stack.orchestrator.config.is_local:
            raise RuntimeError(
                "The implementation of this pipeline "
                "requires that you are running on a local "
                "machine with a running instance of label-studio "
                "configured in the stack as annotator."
                " Please switch to a stack that contains a local "
                "orchestrator and a local label-studio "
                "annotator. See the README for more information "
                "on this setup."
            )

        # Export data from label studio
        data_export_pipeline.with_options(
            config_path="configs/data_export.yaml"
        )()

    if train:
        try:
            client.get_model_version(
                model_name_or_id=ZENML_MODEL_NAME,
                model_version_name_or_number_or_id=ModelStages.STAGING,
            )
        except KeyError:
            raise RuntimeError(
                "This pipeline requires that there is a version of its "
                "associated model in the `STAGING` stage. Make sure you run "
                "the `data_export_pipeline` at least once to create the Model "
                "along with a version of this model. After this you can "
                "promote the version of your choice, either through the "
                "frontend or with the following command: "
                f"`zenml model version update {ZENML_MODEL_NAME} latest "
                f"-s staging`"
            )

        if train_local:
            config_path = "configs/training_pipeline.yaml"
        else:
            config_path = "configs/training_pipeline_remote_gpu.yaml"

        # Train model on data
        training_pipeline.with_options(config_path=config_path)()

    if batch_inference:
        try:
            client.get_model_version(
                model_name_or_id=ZENML_MODEL_NAME,
                model_version_name_or_number_or_id=ModelStages.PRODUCTION,
            )
        except KeyError:
            raise RuntimeError(
                "This pipeline requires that there is a version of its "
                "associated model in the `Production` stage. Make sure you run "
                "the `data_export_pipeline` at least once to create the Model "
                "along with a version of this model. After this you can "
                "promote the version of your choice, either through the "
                "frontend or with the following command: "
                f"`zenml model version update {ZENML_MODEL_NAME} staging "
                f"-s production`"
            )

        inference_pipeline.with_options(
            config_path="configs/inference_pipeline.yaml"
        )()

    if fiftyone:
        import fiftyone as fo

        artifact = Client().get_artifact_version(
            name_id_or_prefix=PREDICTIONS_DATASET_ARTIFACT_NAME
        )
        dataset_json = artifact.load()
        dataset = fo.Dataset.from_json(dataset_json, persistent=False)
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
