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
from typing import Annotated, Any, Dict, Tuple

from ultralytics import YOLO
from zenml import ArtifactConfig, log_artifact_metadata, step

from materializers.label_studio_yolo_dataset_materializer import (
    LabelStudioYOLODataset,
)
from materializers.ultralytics_materializer import UltralyticsMaterializer
from utils.dataset_utils import load_and_split_data


@step(
    output_materializers={"Trained_YOLO": UltralyticsMaterializer},
    enable_cache=True,
)
def train_model(
    epochs: int,
    batch_size: int,
    imgsz: int,
    model: YOLO,
    dataset: LabelStudioYOLODataset,
    data_source: str,
    is_quad_gpu_env: bool = False,
) -> Tuple[
    Annotated[
        YOLO, ArtifactConfig(name="Trained_YOLO", is_model_artifact=True)
    ],
    Annotated[Dict[str, Any], "validation_metrics"],
]:
    """Trains a model on a dataset.

    Args:
        epochs: Number of epochs to train the model for.
        batch_size: Batch size for training
        imgsz: Image size for training
        model: YOLO model to train.
        dataset: Dataset to train the model on.
        data_source: Source where the data lives
        is_quad_gpu_env: Whether we are in an env with 4 gpus

    Returns:
        Tuple[YOLO, Dict[str, Any]]: Trained model and validation metrics.
    """
    data_path = load_and_split_data(dataset=dataset, data_source=data_source)

    if is_quad_gpu_env:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=[0, 1, 2, 3]
        )
    else:
        model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz
        )

    metrics = model.val()  # evaluate model performance on the validation set

    log_artifact_metadata(
        artifact_name="Trained_YOLO",
        metadata={"metrics": metrics.results_dict},
    )

    return model, metrics.results_dict
